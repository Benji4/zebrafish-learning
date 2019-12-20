import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage.filters import percentile_filter
import time
from tqdm import tqdm
import os

def tail_func2(x, mu, sigma, scale, offset):
    return scale * np.exp(-(x-mu)**4/(2.0*sigma**2))**.2 + offset

if __name__ == '__main__':
    start = time.time()

    print("Loading input...")

    filename = '../z_small'
    loaded = np.load(filename + '.npz')
    X, y = loaded['inputs'], loaded['targets']

    print("Elapsed time:",time.time()-start)

    debug = True
    display = True
    direction = 'right'
    plotlengths = False
    offset_left = 8
    sleep_time = 0 # debugging

    out_dir = 'figs'

    height = X[0,0].shape[0]
    start_point = np.array([offset_left, int(height / 2)])  # left-most, vertical center (offset_left,128)

    directions = {"up": [0, -1], "down": [0, 1], "left": [-1, 0], "right": [1, 0]}

    fitted_tails = []

    print("Starting tailfit for all videos in", filename+'.npz')

    i = 0
    error_idxs = []
    for event in tqdm(X):
        try:
            fitted_tail = []


            max_points = 200  # mostly in case it somehow gets stuck in a loop, and to preallocate the result array
            frame_fit = np.zeros((max_points, 2))
            first_frame = True
            widths, convolveresults = [], []
            test, slices = [], []
            for frame in event:

                # The tail fitting procedure is quite sensitive to the contrast of tail, agarose and background.
                # If the variance is lower, there are more dark areas around the fish, especially the agarose.
                # That part is detrimental for tailfitting.
                # Thus, by performing another gamma correction, we help the algorithm to not confuse agarose and tail.
                # variance = np.var(frame)
                # upper, lower = 4.5, 0.5
                # gamma = upper / (1 + np.exp(0.07 * (-variance + 815))) + lower
                # frame = np.array(np.clip(pow(frame / 255.0, gamma) * 255.0, 0, 255), dtype=np.uint8)

                if display:
                    frame_display = frame.copy()
                if direction:
                    guess_vector = np.array(directions[direction])
                else:
                    raise Exception('Need to define a direction!')  # could ask here

                if first_frame:
                    if type(start_point) == type(np.array([])) or type(start_point) is list:
                        current = np.array(start_point)
                        point = current
                    else:
                        raise Exception('Need to define a direction')

                    if frame.ndim == 2:
                        hist = np.histogram(frame[:, :], 10, (0, 255))
                    elif frame.ndim == 3:
                        hist = np.histogram(frame[:, :, 0], 10, (0, 255))
                    else:
                        raise Exception('Unknown video format!')

                    background = hist[1][hist[0].argmax()] / 2 + hist[1][min(hist[0].argmax() + 1, len(hist[0]))] / 2
                    if background < 200: print("Background is unusual. Value:", background)
                    # find background - 10 bin hist of frame, use most common as background
                    if frame.ndim == 2:
                        fish = frame[point[1] - 2:point[1] + 2, point[0] - 2:point[0] + 2].mean()
                    elif frame.ndim == 3:
                        fish = frame[point[1] - 2:point[1] + 2, point[0] - 2:point[0] + 2, 0].mean()
                    # find fish luminosity - area around point
                    # print("Starting tailfit")

                    guess_line_width = 51
                    normpdf = norm.pdf(np.arange((-guess_line_width + 1) / 4 + 1, (guess_line_width - 1) / 4), 0,
                                       8)  # gaussian kernel, used to find middle of tail
                    # if display:
                    #     cv2.namedWindow("frame_display")
                    #     cv2.moveWindow("frame_display", x_window, y_window)


                    # time.sleep(5)

                else:
                    current = fitted_tail[-1][0, :]

                tailpoint_spacing = 5  # change this multiplier to change the point spacing
                for count in range(max_points):

                    if count == 0:
                        guess = current
                    elif count == 1:
                        guess = current + guess_vector * tailpoint_spacing  # can't advance guess vector, since we didn't move from our previous point
                    else:
                        denominator = ((guess_vector ** 2).sum()) ** .5
                        if denominator != 0:
                            guess_vector = guess_vector / (((guess_vector ** 2).sum()) ** .5)  # normalize guess vector
                            guess = current + guess_vector * tailpoint_spacing
                        else:
                            guess = current
                    guess_line_start = guess + np.array([-guess_vector[1], guess_vector[0]]) * guess_line_width / 2
                    guess_line_end = guess + np.array([guess_vector[1], -guess_vector[0]]) * guess_line_width / 2
                    x_indices = np.int_(np.linspace(guess_line_start[0], guess_line_end[0], guess_line_width))
                    y_indices = np.int_(np.linspace(guess_line_start[1], guess_line_end[1], guess_line_width))

                    if max(y_indices) >= frame.shape[0] or min(y_indices) < 0 or max(x_indices) >= frame.shape[1] or min(
                            x_indices) < 0:
                        y_indices = np.clip(y_indices, 0, frame.shape[0] - 1)
                        x_indices = np.clip(x_indices, 0, frame.shape[1] - 1)
                        # if debug:
                        #     print("Tail surpassed edge of the frame, clipping points.")

                    guess_slice = frame[y_indices, x_indices]  # the frame is transposed compared to what might be expected
                    if guess_slice.ndim == 2:
                        guess_slice = guess_slice[:, 0]
                    else:
                        guess_slice = guess_slice[:]

                    if fish < background:
                        guess_slice = (background - guess_slice)
                    else:
                        guess_slice = (guess_slice - background)

                    slices += [guess_slice]
                    hist = np.histogram(guess_slice, 10)
                    max_bin = guess_slice[((hist[1][hist[0].argmax()] <= guess_slice) & (guess_slice < hist[1][hist[0].argmax() + 1]))]
                    guess_slice -= max_bin.mean()
                    # baseline subtraction

                    sguess = percentile_filter(guess_slice, 50,
                                               5)  # this seems to do a nice job of smoothing out while not moving edges too much

                    if first_frame:
                        # first time through, profile the tail
                        tailedges = np.where(np.diff(sguess > (sguess.max() * .25)))[0] # indices of the elements where condition holds true
                        if len(tailedges) >= 2:
                            tailedges = tailedges - len(sguess) / 2.0
                            tailindexes = tailedges[np.argsort(np.abs(tailedges))[0:2]]
                            result_index_new = (tailindexes).mean() + len(sguess) / 2.0
                            widths += [abs(tailindexes[0] - tailindexes[1])]
                            print("Widths:",widths)
                        else:
                            result_index_new = None
                            tail_length = count
                            break
                        results = np.convolve(normpdf, guess_slice, "valid")
                        convolveresults += [results]
                        result_index = results.argmax() - results.size / 2 + guess_slice.size / 2

                        # if debug:
                        #     print(result_index_new)

                        newpoint = np.array([x_indices[int(result_index_new)], y_indices[int(result_index_new)]])

                    else:
                        results = np.convolve(tailfuncs[count], guess_slice, "valid")
                        result_index = results.argmax() - results.size / 2 + guess_slice.size / 2
                        newpoint = np.array([x_indices[int(result_index)], y_indices[int(result_index)]])

                    if first_frame:
                        if count > 10:
                            # @ SCALE FIT GOODNESS WITH CONTRAST
                            trapz = [np.trapz(result - result.mean()) for result in convolveresults]
                            slicesnp = np.vstack(slices)
                            if np.array(trapz[-3:]).mean() < .2:
                                ##                        pdb.set_trace()
                                tail_length = count
                                break

                            elif slicesnp[-1, int(result_index) - 2:int(result_index) + 2].mean() < 10:
                                ##                    elif -np.diff(sliding_average(slicesnp.mean(1)),4).min()<0:

                                ##                    elif np.diff(scipy.ndimage.filters.percentile_filter(trapz,50,4)).min()<-20:
                                ##                        print np.abs(np.diff(trapz))
                                ##                        pdb.set_trace()
                                tail_length = count
                                break
                    ##            elif count > 1 and pylab.trapz(results-results.mean())<.3: #lower means higher contrast threshold
                    elif count > tail_length * .8 and np.power(newpoint - current, 2).sum() ** .5 > tailpoint_spacing * 1.5:
                        ##                print count, ' Point Distance Break', np.power(newpoint-current,2).sum()**.5
                        break
                    elif count == tail_length:
                        break  # should be end of the tail
                    # threshold changes with tail speed?
                    # also could try overfit, and seeing where the elbow is

                    if display:
                        cv2.circle(frame_display, (int(newpoint[0]), int(newpoint[1])), 2, color=(255,255,255))
                    ##                frame_display[y_indices,x_indices]=0

                    frame_fit[count, :] = newpoint

                    if count > 0:
                        guess_vector = newpoint - current
                    current = newpoint

                if first_frame:
                    if not widths:
                        # if widths is empty, look for a good starting point somewhere else
                        print("Issue with video", i)
                        offset_left += 5
                        break
                    swidths = percentile_filter(widths, 50, 8)
                    swidths = np.lib.pad(swidths, [0, 5], mode='edge')
                    tailfuncs = [
                        tail_func2(np.arange((-guess_line_width + 1) / 4 + 1, (guess_line_width - 1) / 4), 0, swidth, 1, 0) for
                        swidth in swidths]

                fitted_tail.append(np.copy(frame_fit[:count]))
                if display:

                    if i==2 and len(fitted_tail)==70:
                        # cv2.putText(frame_display, str(count), (340, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (225, 10, 20));
                        # cv2.putText(frame_display, str(len(fitted_tail) - 1), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        #             (25, 10, 20));  # -1 because the current frame has already been appended
                        # cv2.imshow("frame_display", frame_display)

                        plt.imshow(frame_display, cmap='gray')
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, 'points.pdf'), bbox_inches='tight')

                    if first_frame:
                        delaytime = 1
                    else:
                        minlen = min([fitted_tail[-2].shape[0], fitted_tail[-1].shape[0]]) - 1
                        delaytime = int(min(max(
                            (np.abs((fitted_tail[-2][minlen, :] - fitted_tail[-1][minlen, :]) ** 2).sum() ** .5) ** 1.2 * 3 - 1,
                            1), 500))
                    ##            print delaytime
                    cv2.waitKey(delaytime)

                    if display:
                        time.sleep(sleep_time)

                # if output_jpegs:
                #     if first_frame:
                #         jpegs_dir = pickdir()
                #         if not os.path.exists(jpegs_dir):
                #             os.makedirs(jpegs_dir)
                #     jpg_out = Image.fromarray(frame_display)
                #     jpg_out.save(os.path.normpath(jpegs_dir + '\\' + str(len(fitted_tail) - 1) + '.jpg'))

                first_frame = False
                ##        cap.set(cv2.CAP_PROP_POS_FRAMES,float(len(fitted_tail)) );  #workaround for raw videos crash, but massively (ie 5x) slower


            fit_lengths = np.array([len(i) for i in fitted_tail])
            if np.std(fit_lengths) > 3 or plotlengths:
                print('Abnormal variances in tail length detected, check results')

                # pylab.plot(range(0, len(fitted_tail)), fit_lengths)
                # pylab.ylim((0, 5 + max(fit_lengths)))
                # pylab.xlabel("Frame")
                # pylab.ylabel('Tail points')
                # pylab.title('Tail fit lengths')
                # print('Close graph to continue!')
                # pylab.show()

            if any(fit_lengths < 25):
                print("Issue with video", i)
                print("Warning - short tail detected in some frame with only ", min(fit_lengths), "points.")

            if len(fitted_tail) != len(event):
                print("Issue with video", i)
                print(
                    "Warning - number of frames processed doesn't match number of video frames - can happen with videos over 2gb!")
                print("Frames processed: ", len(fitted_tail))
                print("Actual frames according to video header: ", len(event))

            fitted_tails.append(fitted_tail)
            i+=1
            ### Done with one video ###
        except:
            print("Video", i, "could not be fitted.")
            error_idxs.append(i)
            i+=1


    ### Done with all videos ###

    fitted_tails = np.array(fitted_tails)

    print("All videos fitted in %.2f seconds" % (time.time() - start))


    if display:
        cv2.destroyAllWindows()

    # print(fitted_tails[0])
    # print(fitted_tails[0,0])

    print('Indices of videos with errors:',error_idxs)
    print('fitted_tails.shape:',fitted_tails.shape)
    print('Remember that fitted_tails contains a list of variable length for each frame which is the number of fitted points, thus (n,150,num_points).')


    # save fitted_tails for use in the SVM
    np.savez_compressed(filename + '_tail.npz', tails=fitted_tails, targets=y) # (n,frames,points); points varies
