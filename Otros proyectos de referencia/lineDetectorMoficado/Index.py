from Pipeline import *

from moviepy.editor import VideoFileClip
from IPython.display import HTML

#Demo Camera...

# Note: the calibration process only needs to be run once in the absense of the pickled file
# containing the calculated aforementioned params
if os.path.exists('camera_calib.p'):
    with open('camera_calib.p', mode='rb') as f:
        data = pickle.load(f)
        mtx, dist = data['mtx'], data['dist']
        print('Loaded the saved camera calibration matrix & dist coefficients!')
else:
    mtx, dist = calibrate_camera()
    with open('camera_calib.p', mode='wb') as f:
        pickle.dump({'mtx': mtx, 'dist': dist}, f)


if 1 in plot_demo:
    ccimg = cv2.imread('camera_cal/calibration1.jpg')
    ccimg_undist = undistort(ccimg, mtx, dist)

    plot_images([
        (ccimg, 'Original Image'),
        (ccimg_undist, 'Undistorted Image')
    ])

    img_orig = mpimg.imread(test_img_paths[6])
    img = undistort(img_orig, mtx, dist)

    plot_images([
        (img_orig, 'Original Image'),
        (img, 'Undistorted Image')
    ])

#Demo Persperctive...
# Undistort a sample camera calibration image and a sample test image
if 2 in plot_demo:
    for path in test_img_paths[:]:
        get_image(mtx, dist, path, visualise=True)

#Demo Threshold...
if 3 in plot_demo:
    for img_path in test_img_paths[:2]: #video2[5:10]:
        img, _ = get_image(mtx, dist, img_path)
        get_binary_image(img, visualise=True)

#Demo Histogram...
if 4 in plot_demo:

    cache = np.array([])

    for img_path in test_img_paths[:]: #video2[134:138]:
        img, _ = get_image(mtx, dist, img_path)
        binary = get_binary_image(img, visualise=False)
        polyfit_sliding_window(cache, img, binary, visualise=True) #polyfit_sliding_window(img, binary, visualise=True)

#Demo AdaptativeSearch

if 5 in plot_demo:
    cache = np.array([])
    attempts = 0
    max_attempts = 4
    reset = True

    for frame_path in video2[0: 5]:
        img = mpimg.imread(frame_path)
        warped, (M, invM) = get_image(frame_path)

        if reset == True:
            binary = get_binary_image(warped)
            ret, out, poly_param = polyfit_sliding_window(cache, binary, visualise=False, diagnostics=True)
            if ret:
                reset = False
                cache = np.array([poly_param])

        else:
            out, poly_param = polyfit_adapt_search(warped, poly_param, visualise=False, diagnostics=False)
            if attempts == max_attempts:
                attempts = 0
                reset = True

            out_unwarped = cv2.warpPerspective(out, invM, (IMG_SHAPE[1], IMG_SHAPE[0]), flags=cv2.INTER_LINEAR)

            img_overlay = np.copy(img)
            img_overlay = cv2.addWeighted(out_unwarped, 0.5, img, 0.5, 0)

            plot_images([(warped, 'Original'), (out, 'Out'), (img_overlay, 'Overlay')], figsize=(20, 18))

#Demo Compute Meters y Pixels
if 6 in plot_demo:
    visualise = True
else:
    visualise = False

img, _ = get_image(mtx, dist, test_img_paths[0])
y_mppx1, x_mppx1 = compute_mppx(img, dashed_line_loc='right', visualise=visualise)

img, _ = get_image(mtx, dist, test_img_paths[1])
y_mppx2, x_mppx2 = compute_mppx(img, dashed_line_loc='left', visualise=visualise)

x_mppx = (x_mppx1 + x_mppx2) / 2
y_mppx = (y_mppx1 + y_mppx2) / 2

print('Average meter/px along x-axis: {:.4f}'.format(x_mppx))
print('Average meter/px along y-axis: {:.4f}'.format(y_mppx))

#Demo Line Curvature
if 7 in plot_demo:
    cache = np.array([])

    for img_path in video1[1000: 1010]:  # test_img_paths:
        img = mpimg.imread(img_path)
        warped, (M, invM) = get_image(mtx, dist, img_path)

        binary = get_binary_image(warped)
        ret, img_poly, poly_param = polyfit_sliding_window(binary)

        left_curverad, right_curverad = compute_curvature(poly_param, y_mppx, x_mppx)
        curvature = (left_curverad + right_curverad) / 2
        offset = compute_offset_from_center(poly_param, x_mppx)
        result = draw(img, warped, invM, poly_param, curvature, offset)

        plot_images([(img_poly, 'Polyfit'), (result, 'Result')])
#Demo Pipeline
if 8 in plot_demo:
    print('Demo of consecutive frames')

    cache = np.array([])
    attempts = 0
    reset = True

    for img_path in video2[0:15]:
        img = mpimg.imread(img_path)
        result, reset, poly_param, attempts, cache = pipeline(y_mppx, x_mppx, reset, cache, poly_param, attempts, mtx, dist, img, visualise=True, diagnostics=1)

#Index
poly_param=0;
#result = pipeline(y_mppx, x_mppx, reset, cache, poly_param, attempts, mtx, dist, img, visualise=True, diagnostics=1)
process_frame = lambda frame: pipeline(y_mppx, x_mppx, reset, cache, poly_param, attempts, mtx, dist, frame, diagnostics=1)

# Pipeline initialisation
cache = np.array([])
attempts = 0
reset = True

video_output = 'project_video_output.mp4'
video_input = VideoFileClip('Videos/20190927_190235.mp4')#.subclip(18,27)
processed_video = video_input.fl_image(process_frame)
processed_video.write_videofile(video_output, audio=False)

