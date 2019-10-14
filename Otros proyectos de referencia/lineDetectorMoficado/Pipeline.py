from LineCurvature import *
def triple_split_view(images):
    '''
    Utility function to create triple split view for display in video
    :param images ([ndarray]): List of images
    :returm (ndarray): Single RGB image
    '''

    scale_factor = 2

    # Sizes/shapes are in (x,y) format for convenience use with cv2
    img_shape = IMG_SHAPE[::-1]
    scaled_size = (round(img_shape[0] / scale_factor), round(img_shape[1] / scale_factor))
    x_max, y_max = img_shape[0], img_shape[1] + scaled_size[1]  # x, y + y'

    # Top-left corner positions for each of the three windows
    positions = [(0, 0), (0, img_shape[1]), (round(0.5 * img_shape[0]), img_shape[1])]
    sizes = [img_shape, scaled_size, scaled_size]

    out = np.zeros((y_max, x_max, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        # Resize the image
        if img.shape[0] != sizes[idx][1] | img.shape[1] != sizes[idx][0]:
            img = cv2.resize(img, dsize=sizes[idx])

        # Place the resized image onto the final output image
        x, y = positions[idx]
        w, h = sizes[idx]
        out[y:min(y + h, y_max), x:min(x + w, x_max), :] = img[:min(h, y_max - y), :min(w, x_max - x)]

    return out


def pipeline(y_mppx, x_mppx, reset, cache, poly_param, attempts, mtx, dist, img, visualise=False, diagnostics=False):
    #global cache
    #global poly_param  # Important for successive calls to the pipeline
    #global attempts
    #global reset
    max_attempts = 5

    result = np.copy(img)
    warped, (M, invM) = preprocess_image(mtx, dist, img)
    title = ''

    #try:
    if reset == True:
        title = 'Sliding window'
        if diagnostics: print(title)

        binary = get_binary_image(warped)
        ret, img_poly, poly_param = polyfit_sliding_window(cache, img, binary, diagnostics=diagnostics)
        if ret:
            if diagnostics: print('Success!')
            reset = False
            cache = np.array([poly_param])
        else:
            if len(img_poly) == 0:
                print('Sliding window failed!')
                return img

    else:
        title = 'Adaptive Search'
        if diagnostics: print(title)

        img_poly, poly_param = polyfit_adapt_search(warped, poly_param, diagnostics=diagnostics)
        if attempts == max_attempts:
            if diagnostics: print('Resetting...')
            reset = True
            attempts = 0

    left_curverad, right_curverad = compute_curvature(poly_param, y_mppx, x_mppx)
    offset = compute_offset_from_center(poly_param, x_mppx)
    result = draw(mtx, dist, img, warped, invM, poly_param, (left_curverad + right_curverad) / 2, offset)

    blended_warped_poly = cv2.addWeighted(img_poly, 0.6, warped, 1, 0)
    ret2 = np.hstack([img_poly, blended_warped_poly])
    ret3 = np.hstack([result, warped])
    #         ret3 = triple_split_view([result, img_poly, blended_warped_poly])
    ret3 = np.vstack([ret3, ret2])
    if visualise:
        plt.figure(figsize=(20, 12))
        plt.title(title)
        plt.imshow(ret3)

    return ret3

    #except Exception as e:
     #   print(e)
      #  return img

    # -----------------------------------------------------------------------------------------
# DEMO
'''
if 8 in plot_demo:
    print('Demo of consecutive frames')

    cache = np.array([])
    attempts = 0
    reset = True

    for img_path in video2[0:15]:
        img = mpimg.imread(img_path)
        result = pipeline(img, visualise=True, diagnostics=1)
'''