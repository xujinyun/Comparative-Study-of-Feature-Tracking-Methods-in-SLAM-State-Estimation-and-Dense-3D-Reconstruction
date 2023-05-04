/*
This code is the implementation of our paper "R3LIVE: A Robust, Real-time, RGB-colored,
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored,
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package."
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping."
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include "rgbmap_tracker.hpp"
#include <iostream>
#include <fstream>

Rgbmap_tracker::Rgbmap_tracker()
{
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.05);
    if (m_lk_optical_flow_kernel == nullptr)
    {
        m_lk_optical_flow_kernel = std::make_shared<LK_optical_flow_kernel>(cv::Size(21, 21), 3, criteria,
                                                                            cv_OPTFLOW_LK_GET_MIN_EIGENVALS);
    }
}

void Rgbmap_tracker::update_and_append_track_pts(std::shared_ptr<Image_frame> &img_pose, Global_map &map_rgb,
                                                 double mini_dis, int minimum_frame_diff)
{
    Common_tools::Timer tim;
    tim.tic();
    double u_d, v_d;
    int u_i, v_i;
    double max_allow_repro_err = 2.0 * img_pose->m_img_cols / 320.0;
    Hash_map_2d<int, float> map_2d_pts_occupied;

    for (auto it = m_map_rgb_pts_in_last_frame_pos.begin(); it != m_map_rgb_pts_in_last_frame_pos.end();)
    {
        RGB_pts *rgb_pt = ((RGB_pts *)it->first);
        vec_3 pt_3d = ((RGB_pts *)it->first)->get_pos();
        int res = img_pose->project_3d_point_in_this_img(pt_3d, u_d, v_d, nullptr, 1.0);
        u_i = std::round(u_d / mini_dis) * mini_dis;
        v_i = std::round(v_d / mini_dis) * mini_dis;

        double error = vec_2(u_d - it->second.x, v_d - it->second.y).norm();

        if (error > max_allow_repro_err)
        {
            // cout << "Remove: " << vec_2(it->second.x, it->second.y).transpose() << " | " << vec_2(u, v).transpose()
            // << endl;
            rgb_pt->m_is_out_lier_count++;
            if (rgb_pt->m_is_out_lier_count > 1 || (error > max_allow_repro_err * 2))
            // if (rgb_pt->m_is_out_lier_count > 3)
            {
                rgb_pt->m_is_out_lier_count = 0; // Reset
                it = m_map_rgb_pts_in_last_frame_pos.erase(it);
                continue;
            }
        }
        else
        {
            rgb_pt->m_is_out_lier_count = 0;
        }

        if (res)
        {
            double depth = (pt_3d - img_pose->m_pose_w2c_t).norm();
            if (map_2d_pts_occupied.if_exist(u_i, v_i) == false)
            {
                map_2d_pts_occupied.insert(u_i, v_i, depth);
                // it->second = cv::Point2f(u, v);
            }
        }
        else
        {
            // m_map_rgb_pts_in_last_frame_pos.erase(it);
        }
        it++;
    }

    int new_added_pts = 0;

    tim.tic("Add");
    while (map_rgb.m_updated_frame_index < img_pose->m_frame_idx - minimum_frame_diff)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
    }
    map_rgb.m_mutex_pts_vec->lock();
    int new_add_pt = 0;

    if (map_rgb.m_pts_rgb_vec_for_projection != nullptr)
    {
        int pt_size = map_rgb.m_pts_rgb_vec_for_projection->size();
        for (int i = 0; i < pt_size; i++)
        {
            if (m_map_rgb_pts_in_last_frame_pos.find((*map_rgb.m_pts_rgb_vec_for_projection)[i].get()) !=
                m_map_rgb_pts_in_last_frame_pos.end())
            {
                continue;
            }
            vec_3 pt_3d = (*map_rgb.m_pts_rgb_vec_for_projection)[i]->get_pos();
            int res = img_pose->project_3d_point_in_this_img(pt_3d, u_d, v_d, nullptr, 1.0);
            u_i = std::round(u_d / mini_dis) * mini_dis;
            v_i = std::round(v_d / mini_dis) * mini_dis;
            // vec_3 rgb_color = img_pose->get_rgb(u, v);
            // double grey = img_pose->get_grey_color(u, v);
            // (*map_rgb.m_pts_rgb_vec_for_projection)[i]->update_gray(grey);
            // (*map_rgb.m_pts_rgb_vec_for_projection)[i]->update_rgb(rgb_color);
            if (res)
            {
                double depth = (pt_3d - img_pose->m_pose_w2c_t).norm();
                if (map_2d_pts_occupied.if_exist(u_i, v_i) == false)
                {
                    map_2d_pts_occupied.insert(u_i, v_i, depth);
                    m_map_rgb_pts_in_last_frame_pos[(*map_rgb.m_pts_rgb_vec_for_projection)[i].get()] =
                        cv::Point2f(u_d, v_d);
                    new_added_pts++;
                }
            }
            new_add_pt++;
            if (m_map_rgb_pts_in_last_frame_pos.size() >= m_maximum_vio_tracked_pts)
            {
                break;
            }
        }
        // cout << "Tracker new added pts = " << new_added_pts << " |  " << map_rgb.m_pts_rgb_vec_for_projection->size()
        //      << " | " << img_pose->m_frame_idx << " | " << map_rgb.m_updated_frame_index
        //      << ", cost = " << tim.toc("Add") << endl;
    }

    map_rgb.m_mutex_pts_vec->unlock();
    // cout << "update tracking vecotr 168" << endl;
    update_last_tracking_vector_and_ids();
    // cout << "Update points cost time = " << tim.toc() << endl;
}

void Rgbmap_tracker::reject_error_tracking_pts(std::shared_ptr<Image_frame> &img_pose, double dis)
{
    double u, v;
    int remove_count = 0;
    int total_count = m_map_rgb_pts_in_current_frame_pos.size();
    // cout << "Cam mat: " <<img_pose->m_cam_K << endl;
    // cout << "Image pose: ";
    // img_pose->display_pose();
    scope_color(ANSI_COLOR_BLUE_BOLD);
    for (auto it = m_map_rgb_pts_in_current_frame_pos.begin(); it != m_map_rgb_pts_in_current_frame_pos.end(); it++)
    {
        cv::Point2f predicted_pt = it->second;
        vec_3 pt_3d = ((RGB_pts *)it->first)->get_pos();
        int res = img_pose->project_3d_point_in_this_img(pt_3d, u, v, nullptr, 1.0);
        if (res)
        {
            if ((fabs(u - predicted_pt.x) > dis) || (fabs(v - predicted_pt.y) > dis))
            {
                // Remove tracking pts
                m_map_rgb_pts_in_current_frame_pos.erase(it);
                remove_count++;
            }
        }
        else
        {
            // cout << pt_3d.transpose() << " | ";
            // cout << "Predicted: " << vec_2(predicted_pt.x, predicted_pt.y).transpose() << ", measure: " << vec_2(u,
            // v).transpose() << endl;
            m_map_rgb_pts_in_current_frame_pos.erase(it);
            remove_count++;
        }
    }
    cout << "Total pts = " << total_count << ", rejected pts = " << remove_count << endl;
}

// inline void image_equalize(cv::Mat &img, int amp)
// {
//     cv::Mat img_temp;
//     cv::Size eqa_img_size = cv::Size(std::max(img.cols * 32.0 / 640, 4.0), std::max(img.cols * 32.0 / 640, 4.0));
//     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_img_size);
//     // Equalize gray image.
//     clahe->apply(img, img_temp);
//     img = img_temp;
// }

// inline cv::Mat equalize_color_image_ycrcb(cv::Mat &image)
// {
//     cv::Mat hist_equalized_image;
//     cv::cvtColor(image, hist_equalized_image, cv::COLOR_BGR2YCrCb);

//     //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
//     std::vector<cv::Mat> vec_channels;
//     cv::split(hist_equalized_image, vec_channels);

//     //Equalize the histogram of only the Y channel
//     // cv::equalizeHist(vec_channels[0], vec_channels[0]);
//     image_equalize( vec_channels[0], 2 );
//     cv::merge(vec_channels, hist_equalized_image);
//     cv::cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);
//     return hist_equalized_image;
// }

void Rgbmap_tracker::track_img_R3live(std::shared_ptr<Image_frame> &img_pose, double dis, int if_use_opencv)
{
    Common_tools::Timer tim;
    m_current_frame = img_pose->m_img; // 取图像
    m_current_frame_time = img_pose->m_timestamp;

    m_map_rgb_pts_in_current_frame_pos.clear(); // 当前帧跟踪到的地图点
    if (m_current_frame.empty())                // 检查图像是否为空
        return;
    cv::Mat frame_gray = img_pose->m_img_gray; // 取灰度图     not required
    tim.tic("HE");
    tim.tic("opTrack");
    std::vector<uchar> status;
    std::vector<float> err;
    // 取上一帧跟踪的像素点，并检查数量是否足够
    m_current_tracked_pts = m_last_tracked_pts;
    int before_track = m_last_tracked_pts.size();
    if (m_last_tracked_pts.size() < 30)
    {
        m_last_frame_time = m_current_frame_time;
        return;
    }
    // cout << "before LK = " << m_last_tracked_pts.size() << endl;
    // cout << "before LK = " << m_old_ids.size() << endl;
    // cout << "before LK = " << m_current_tracked_pts.size() << endl;
    // ////////////////////////////////////////////////////////////////////////////////////////////
    // // 调用LK_optical_flow_kernel::track_image，光流跟踪，输出跟踪后的像素点m_current_tracked_pts
    m_lk_optical_flow_kernel->track_image(frame_gray, m_last_tracked_pts, m_current_tracked_pts, status, 2);
    // ////////////////////////////////////////////////////////////////////////////////////////////
    // cout << "after LK = " << m_last_tracked_pts.size() << endl;
    // cout << "after LK = " << m_old_ids.size() << endl;
    // cout << "after LK = " << m_current_tracked_pts.size() << endl;
    // // 根据跟踪的结果，对容器进行裁减
    reduce_vector(m_last_tracked_pts, status);    // 成功跟踪的上一帧的点
    reduce_vector(m_old_ids, status);             // 成功跟踪的上一帧像素点所对应的地图点idx
    reduce_vector(m_current_tracked_pts, status); // 当前帧成功跟踪的点

    int after_track = m_last_tracked_pts.size();

    cv::Mat mat_F;

    tim.tic("Reject_F");
    // 求基础矩阵F
    unsigned int pts_before_F = m_last_tracked_pts.size();
    // can really eliminate moving objects????????????????????????????????????????????????????????????????
    // TODO:
    mat_F = cv::findFundamentalMat(m_last_tracked_pts, m_current_tracked_pts, cv::FM_RANSAC, 1.0, 0.997, status);
    unsigned int size_a = m_current_tracked_pts.size();
    // 根据求解F矩阵的RANSAC结果，去除outliers

    reduce_vector(m_last_tracked_pts, status);
    reduce_vector(m_old_ids, status);
    reduce_vector(m_current_tracked_pts, status);
    // cout << "297 Size after update = " << m_last_tracked_pts.size() << " " << m_current_tracked_pts.size() << " " << m_old_ids.size() << endl;

    m_map_rgb_pts_in_current_frame_pos.clear();
    // 距离上一次跟踪的时间
    double frame_time_diff = (m_current_frame_time - m_last_frame_time);
    // 遍历跟踪成功的点，保存像素点坐标以及跟踪到的速度
    for (uint i = 0; i < m_last_tracked_pts.size(); i++)
    {
        // 用于跳过靠近图像边缘的点
        if (img_pose->if_2d_points_available(m_current_tracked_pts[i].x, m_current_tracked_pts[i].y, 1.0, 0.05))
        {
            // m_rgb_pts_ptr_vec_in_last_frame[ m_old_ids[ i ] ]表示索引为 i 的地图点的指针
            // 这里将地图点转化为RGB_pts指针
            RGB_pts *rgb_pts_ptr = ((RGB_pts *)m_rgb_pts_ptr_vec_in_last_frame[m_old_ids[i]]);
            // 保存当前帧跟踪到的地图点
            m_map_rgb_pts_in_current_frame_pos[rgb_pts_ptr] = m_current_tracked_pts[i];
            // / 计算像素点速度
            cv::Point2f pt_img_vel = (m_current_tracked_pts[i] - m_last_tracked_pts[i]) / frame_time_diff;
            // 保存数据到地图点
            // 成功跟踪的上一帧点
            rgb_pts_ptr->m_img_pt_in_last_frame = vec_2(m_last_tracked_pts[i].x, m_last_tracked_pts[i].y);
            // 成功跟踪的当前帧
            rgb_pts_ptr->m_img_pt_in_current_frame = vec_2(m_current_tracked_pts[i].x, m_current_tracked_pts[i].y);
            rgb_pts_ptr->m_img_vel = vec_2(pt_img_vel.x, pt_img_vel.y);
        }
    }
    /////////Try this
    if (dis > 0)
    {
        reject_error_tracking_pts(img_pose, dis);
    }
    // 保存图像帧
    m_old_gray = frame_gray.clone();
    m_old_frame = m_current_frame;
    // 保存当前帧跟踪到的地图点
    m_map_rgb_pts_in_last_frame_pos = m_map_rgb_pts_in_current_frame_pos;
    // 遍历当前帧跟踪到的地图点,更新如下容器:
    // - m_last_tracked_pts 成功跟踪的上一帧点
    // - m_rgb_pts_ptr_vec_in_last_frame 成功跟踪的上一帧点对应的地图点容器
    // - m_colors
    // - m_old_ids 成功跟踪的上一帧点对应的地图点索引
    // cout << "Update inside the img_tracker 332" << endl;
    update_last_tracking_vector_and_ids();

    m_frame_idx++;
    m_last_frame_time = m_current_frame_time;
}

void Rgbmap_tracker::track_img(std::shared_ptr<Image_frame> &img_pose, double dis, int if_use_opencv)
{
    Common_tools::Timer tim;
    m_current_frame = img_pose->m_img; // 取图像
    m_current_frame_time = img_pose->m_timestamp;

    ////
    // Print the time stamp of current frame

    // read out key points from the corresponding txt file with the same name
    // convert current frame time to string
    cout << std::fixed << "Current frame time = " << m_current_frame_time << endl;

    std::string txt_file_name = "/home/jinyun/R3live/src/R3live_data/degenerate_seq_00" + std::__cxx11::to_string(m_current_frame_time) + ".jpg.txt";
    // cout << "txt_file_name = " << txt_file_name << endl;
    // sleep( 10 );
    std::ifstream input_file(txt_file_name); // open input file
    if (!input_file.is_open())
    {
        std::cerr << "Error: failed to open file." << std::endl;
        return;
    }

    m_current_tracked_pts = m_last_tracked_pts;
    int before_track = m_last_tracked_pts.size();
    tim.tic("HE");
    tim.tic("opTrack");
    std::vector<uchar> status;
    std::vector<float> err;
    if (m_last_tracked_pts.size() < 30)
    {
        m_last_frame_time = m_current_frame_time;
        return;
    }

    std::vector<cv::Point2f> kpts0, kpts1;
    double x0, y0, x1, y1;
    while (input_file >> x0 >> y0 >> x1 >> y1) // read values from file
    {
        kpts0.emplace_back(x0, y0); // add point to kpts0 vector
        kpts1.emplace_back(x1, y1); // add point to kpts1 vector
    }
    input_file.close(); // close input file

    cout << "Get the key points from the txt file" << endl;
    ////////////////////////////////////////////////////////////////////////////////////////

    cout << "before kp = " << m_last_tracked_pts.size() << endl;
    cout << "before kp = " << m_old_ids.size() << endl;
    cout << "before kp = " << m_current_tracked_pts.size() << endl;

    int n = m_old_ids.size();
    int m = kpts0.size();
    if (m > n)
    {
        // Find the corresponding point_num points in kpts0 which have nearest distance to the points in m_last_tracked_pts
        std::vector<int> selected_ids;
        std::vector<cv::Point2f> selected_kpts0, selected_kpts1;
        for (int i = 0; i < n; i++)
        {
            double min_dis = 1000000;
            int min_id = -1;
            for (int j = 0; j < m; j++)
            {   
                // Check whether j is in selected_ids already
                for (int k = 0; k < selected_ids.size(); k++)
                {
                    if (j == selected_ids[k])
                    {
                        continue;
                    }
                }
                double dis = sqrt((kpts0[j].x - m_last_tracked_pts[i].x) * (kpts0[j].x - m_last_tracked_pts[i].x) + (kpts0[j].y - m_last_tracked_pts[i].y) * (kpts0[j].y - m_last_tracked_pts[i].y));
                if (dis < min_dis)
                {
                    min_dis = dis;
                    min_id = j;
                }
            }
            selected_ids.push_back(min_id);
            selected_kpts0.push_back(kpts0[min_id]);
            selected_kpts1.push_back(kpts1[min_id]);
        }
        m_last_tracked_pts = selected_kpts0;
        m_current_tracked_pts = selected_kpts1;
        
    }
    else
    {
        m_last_tracked_pts = kpts0;
        m_current_tracked_pts = kpts1;
    }
    // Find the corresponding point_num points in kpts0 which have nearest distance to the points in m_last_tracked_pts
    cout << "after kp = " << m_last_tracked_pts.size() << endl;
    cout << "after kp = " << m_old_ids.size() << endl;
    cout << "after kp = " << m_current_tracked_pts.size() << endl;

    // exit(0);

    // // Randoomly select n pairs of points from kpts0 and kpts1 where n is the size of m_old_ids
    // std::srand(std::time(nullptr));
    // // Shuffle the indices of the keypoints
    // std::vector<int> indices(m_old_ids.size());
    // std::iota(indices.begin(), indices.end(), 0);
    // std::random_shuffle(indices.begin(), indices.end());
    // std::vector<cv::Point2f> selected_kpts0, selected_kpts1;
    // // Select the first n keypoints from the shuffled list
    // for (int i = 0; i < m_old_ids.size(); i++) {
    //     // cout << "Random index: " << indices[i] << endl;
    //     selected_kpts0.push_back(kpts0[indices[i]]);
    //     selected_kpts1.push_back(kpts1[indices[i]]);
    // }

    // m_last_tracked_pts = selected_kpts0;
    // m_current_tracked_pts = selected_kpts1;

    // cout << "after kp = " << m_last_tracked_pts.size() << endl;
    // cout << "after kp = " << m_old_ids.size() << endl;
    // cout << "after kp = " << m_current_tracked_pts.size() << endl;
    // exit(0);
    // ////////////////////////////////////////////////////////////////////////////////////////////
    // // 调用LK_optical_flow_kernel::track_image，光流跟踪，输出跟踪后的像素点m_current_tracked_pts
    // m_lk_optical_flow_kernel->track_image(frame_gray, m_last_tracked_pts, m_current_tracked_pts, status, 2);
    // ////////////////////////////////////////////////////////////////////////////////////////////
    // // 根据跟踪的结果，对容器进行裁减
    // reduce_vector(m_last_tracked_pts, status);    // 成功跟踪的上一帧的点
    // reduce_vector(m_old_ids, status);             // 成功跟踪的上一帧像素点所对应的地图点idx
    // reduce_vector(m_current_tracked_pts, status); // 当前帧成功跟踪的点

    int after_track = m_last_tracked_pts.size();

    cv::Mat mat_F;

    tim.tic("Reject_F");
    // 求基础矩阵F
    mat_F = cv::findFundamentalMat(m_last_tracked_pts, m_current_tracked_pts, cv::FM_RANSAC, 1.0, 0.997, status);
    // cout << "mat_F = " << mat_F << endl;
    // cout<<"after_track = "<<after_track<<endl;
    // unsigned int pts_before_F = m_last_tracked_pts.size();
    // // can really eliminate moving objects????????????????????????????????????????????????????????????????
    // // TODO:
    // mat_F = cv::findFundamentalMat(m_last_tracked_pts, m_current_tracked_pts, cv::FM_RANSAC, 1.0, 0.997, status);
    // unsigned int size_a = m_current_tracked_pts.size();
    // 根据求解F矩阵的RANSAC结果，去除outlier
    // cout << "before F = " << m_last_tracked_pts.size() << endl;
    // cout << "before F = " << m_old_ids.size() << endl;
    // cout << "before F = " << m_current_tracked_pts.size() << endl;

    reduce_vector(m_last_tracked_pts, status);
    reduce_vector(m_old_ids, status);
    reduce_vector(m_current_tracked_pts, status);

    // cout << "after F = " << m_last_tracked_pts.size() << endl;
    // cout << "after F = " << m_old_ids.size() << endl;
    // cout << "after F = " << m_current_tracked_pts.size() << endl;

    // sleep(10);

    m_map_rgb_pts_in_current_frame_pos.clear();
    // cout << "Clear done" << endl;
    // 距离上一次跟踪的时间
    double frame_time_diff = (m_current_frame_time - m_last_frame_time);
    // 遍历跟踪成功的点，保存像素点坐标以及跟踪到的速度

    for (uint i = 0; i < m_last_tracked_pts.size(); i++)
    {
        // cout << "Enter loop: " << i << endl;
        // 用于跳过靠近图像边缘的点
        if (img_pose->if_2d_points_available(m_current_tracked_pts[i].x, m_current_tracked_pts[i].y, 1.0, 0.05))
        {
            // m_rgb_pts_ptr_vec_in_last_frame[ m_old_ids[ i ] ]表示索引为 i 的地图点的指针
            // 这里将地图点转化为RGB_pts指针
            // cout << "size of container = " << m_rgb_pts_ptr_vec_in_last_frame.size() << endl;
            // cout << "m_old_ids = " << m_old_ids[i] << endl;
            if (m_old_ids[i] >= m_rgb_pts_ptr_vec_in_last_frame.size())
            {
                continue;
                cout << "outof bound" << endl;
                sleep(10);
            }

            RGB_pts *rgb_pts_ptr = ((RGB_pts *)m_rgb_pts_ptr_vec_in_last_frame[m_old_ids[i]]);
            // 保存当前帧跟踪到的地图点
            m_map_rgb_pts_in_current_frame_pos[rgb_pts_ptr] = m_current_tracked_pts[i];
            // / 计算像素点速度
            cv::Point2f pt_img_vel = (m_current_tracked_pts[i] - m_last_tracked_pts[i]) / frame_time_diff;
            // 保存数据到地图点
            // 成功跟踪的上一帧点
            if (!rgb_pts_ptr)
            {
                cout << "rgb_pts_ptr is NULL" << endl;
            }
            rgb_pts_ptr->m_img_pt_in_last_frame = vec_2(m_last_tracked_pts[i].x, m_last_tracked_pts[i].y);

            // 成功跟踪的当前帧
            rgb_pts_ptr->m_img_pt_in_current_frame = vec_2(m_current_tracked_pts[i].x, m_current_tracked_pts[i].y);
            rgb_pts_ptr->m_img_vel = vec_2(pt_img_vel.x, pt_img_vel.y);
        }
    }
    /////////Try this
    if (dis > 0)
    {
        reject_error_tracking_pts(img_pose, dis);
    }
    // 保存图像帧
    m_old_gray = frame_gray.clone();
    m_old_frame = m_current_frame;
    // 保存当前帧跟踪到的地图点
    m_map_rgb_pts_in_last_frame_pos = m_map_rgb_pts_in_current_frame_pos;
    // 遍历当前帧跟踪到的地图点,更新如下容器:
    // - m_last_tracked_pts 成功跟踪的上一帧点
    // - m_rgb_pts_ptr_vec_in_last_frame 成功跟踪的上一帧点对应的地图点容器
    // - m_colors
    // - m_old_ids 成功跟踪的上一帧点对应的地图点索引

    update_last_tracking_vector_and_ids();

    m_frame_idx++;
    m_last_frame_time = m_current_frame_time;
}

int Rgbmap_tracker::get_all_tracked_pts(std::vector<std::vector<cv::Point2f>> *img_pt_vec)
{
    int hit_count = 0;
    for (auto it = m_map_id_pts_vec.begin(); it != m_map_id_pts_vec.end(); it++)
    {
        if (it->second.size() == m_frame_idx)
        {
            hit_count++;
            if (img_pt_vec)
            {
                img_pt_vec->push_back(it->second);
            }
        }
    }
    cout << "Total frame " << m_frame_idx;
    cout << ", success tracked points = " << hit_count << endl;
    return hit_count;
}

int Rgbmap_tracker::remove_outlier_using_ransac_pnp(std::shared_ptr<Image_frame> &img_pose, int if_remove_ourlier)
{
    Common_tools::Timer tim;
    tim.tic();

    cv::Mat r_vec, t_vec;
    cv::Mat R_mat;
    vec_3 eigen_r_vec, eigen_t_vec;
    std::vector<cv::Point3f> pt_3d_vec, pt_3d_vec_selected;
    std::vector<cv::Point2f> pt_2d_vec, pt_2d_vec_selected;
    std::vector<void *> map_ptr_vec;
    for (auto it = m_map_rgb_pts_in_current_frame_pos.begin(); it != m_map_rgb_pts_in_current_frame_pos.end(); it++)
    {
        map_ptr_vec.push_back(it->first);
        vec_3 pt_3d = ((RGB_pts *)it->first)->get_pos();
        pt_3d_vec.push_back(cv::Point3f(pt_3d(0), pt_3d(1), pt_3d(2)));
        pt_2d_vec.push_back(it->second);
    }
    if (pt_3d_vec.size() < 10)
    {
        return 0;
    }
    if (1)
    {
        std::vector<int> status;
        try
        {
            cv::solvePnPRansac(pt_3d_vec, pt_2d_vec, m_intrinsic, cv::Mat(), r_vec, t_vec, false, 200, 1.5, 0.99,
                               status); // SOLVEPNP_ITERATIVE
        }
        catch (cv::Exception &e)
        {
            scope_color(ANSI_COLOR_RED_BOLD);
            cout << "Catching a cv exception: " << e.msg << endl;
            return 0;
        }
        if (if_remove_ourlier)
        {
            // Remove outlier
            m_map_rgb_pts_in_last_frame_pos.clear();
            m_map_rgb_pts_in_current_frame_pos.clear();
            for (unsigned int i = 0; i < status.size(); i++)
            {
                int inlier_idx = status[i];
                {
                    m_map_rgb_pts_in_last_frame_pos[map_ptr_vec[inlier_idx]] = pt_2d_vec[inlier_idx];
                    m_map_rgb_pts_in_current_frame_pos[map_ptr_vec[inlier_idx]] = pt_2d_vec[inlier_idx];
                }
            }
        }
        // cout << "update tracking vecotr and ids 575" << endl;
        update_last_tracking_vector_and_ids();
    }

    // cv::solvePnP(pt_3d_vec, pt_2d_vec, m_intrinsic, m_dist_coeffs * 0, r_vec, t_vec);

    cv::cv2eigen(r_vec, eigen_r_vec);
    cv::cv2eigen(t_vec, eigen_t_vec);
    // eigen_q solver_q = Sophus::SO3d::exp(eigen_r_vec).unit_quaternion().inverse();
    eigen_q solver_q = Sophus::SO3d::exp(eigen_r_vec).unit_quaternion().inverse();
    vec_3 solver_t = (solver_q * eigen_t_vec) * -1.0;
    // cout << "Solve pose: " << solver_q.coeffs().transpose() << " | " << solver_t.transpose() << endl;
    int if_update = 1;
    double t_diff = (solver_t - img_pose->m_pose_w2c_t).norm();
    double r_diff = (solver_q).angularDistance(img_pose->m_pose_w2c_q) * 57.3;

    if_update = 1;
    t_last_estimated = solver_t;
    if (if_update)
    {

        img_pose->m_pnp_pose_w2c_q = solver_q;
        img_pose->m_pnp_pose_w2c_t = solver_t;

        img_pose->m_pose_w2c_q = solver_q;
        img_pose->m_pose_w2c_t = solver_t;
    }
    else
    {
        img_pose->m_pnp_pose_w2c_q = img_pose->m_pose_w2c_q;
        img_pose->m_pnp_pose_w2c_t = img_pose->m_pose_w2c_t;
    }
    img_pose->refresh_pose_for_projection();
    img_pose->m_have_solved_pnp = 1;
    // cout << "Estimate pose cost time = " << tim.toc() << endl;
    return if_update;
}
