from detector import *


import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import argparse
import cv2
from sort import Sort
from visualize import Visualize

def main():
    args = parse_args()
    display = args.display
    use_dlibTracker  = args.use_dlibTracker
    saver = args.saver
    writer_open_pose_tracked = skvideo.io.FFmpegWriter(
               "/home/payam/project/pytorch_Realtime_Multi-Person_Pose_Estimation/data/tracked_out.mp4",
                inputdict = {
                    '-r': str(20)
                },
                outputdict = {
                    '-r': str(20)
                }
            )

    total_time = 0.0
    total_frames = 0

    # for disp
    if display:
        colours = np.random.rand(32, 3)  # used only for display
        plt.ion()
        fig = plt.figure()

    vis = Visualize()
    if not os.path.exists('output'):
        os.makedirs('output')
    out_file = 'output/townCentreOut.top'

    #init detector
    # detector = GroundTruthDetections()

    #init tracker
    tracker =  Sort(use_dlib= use_dlibTracker) #create instance of the SORT tracker

    if use_dlibTracker:
        print ("Dlib Correlation tracker activated!")
    else:
        print ("Kalman tracker activated!")
    video_capture = cv2.VideoCapture("/local_home/project/Tracking-with-darkflow/data/AVG-TownCentre.mp4")
    _, img = video_capture.read()
    image_shape = img.shape
    lines = np.zeros(image_shape, np.uint8)

    with open(out_file, 'w') as f_out:

        while video_capture.isOpened():
            # get detections

            people = np.zeros(image_shape, np.uint8)
            total_frames +=1
            # fn = 'test/Pictures%d.jpg' % (frame + 1)  # video frames are extracted to 'test/Pictures%d.jpg' with ffmpeg
            # img = io.imread(fn)
            _, img = video_capture.read()
            # img = cv2.resize(img, (0, 0), fx=1.4, fy=1.4)
            if img is None:
                break
            canvas, track_list = handle_one(img)
            detections = np.asarray([np.asarray(track.get_bounding_box()) for track in track_list])
            if (display):
                ax1 = fig.add_subplot(111, aspect='equal')
                ax1.imshow(img)
                if(use_dlibTracker):
                    plt.title('Dlib Correlation Tracker')
                else:
                    plt.title('Kalman Tracker')

            start_time = time.time()
            #update tracker
            trackers = tracker.update(detections,img)

            cycle_time = time.time() - start_time
            total_time += cycle_time

            print('frame: %d...took: %3fs'%(total_frames,cycle_time))
            vis.add_points(trackers)
            for d in trackers:
                f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], total_frames, 1, 1, d[0], d[1], d[2], d[3]))
                if (display):
                    d = d.astype(np.int32)
                    lines = vis.draw_line(d[4], lines)
                    ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                    ec=colours[d[4] % 32, :]))
                    vis.rectangle(people, (d[2], d[3]), (d[0], d[1]),d[4])
                    vis.put_text(people, '%d' % (d[4]), (d[0], d[1]))
                    ax1.set_adjustable('box-forced')
                    #label
                    ax1.annotate('id = %d' % (d[4]), xy=(d[0], d[1]), xytext=(d[0], d[1]))
                    if detections != []:#detector is active in this frame
                        ax1.annotate(" DETECTOR", xy=(5, 45), xytext=(5, 45))

            if (display):
                combined = cv2.addWeighted(canvas, 0.4, people, 0.6, 0)
                combined = cv2.addWeighted(combined, 0.7, lines, 0.3, 0)
                cv2.imshow("lines", combined)
                cv2.waitKey(1)

                plt.axis('off')
                fig.canvas.flush_events()
                plt.draw()
                fig.tight_layout()

                # cv2.imshow("my", data)
                # cv2.waitKey(2)
                #save the frame with tracking boxes
                cv2.imshow("test", canvas)
                cv2.waitKey(1)
                if(saver):
                    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                    writer_open_pose_tracked.writeFrame(combined)
                #         fig.savefig("./frameOut/f%d.jpg"%(total_frames+1),dpi = 200)
                ax1.cla()
    writer_open_pose_tracked.close()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Experimenting Trackers with SORT')
    parser.add_argument('--NoDisplay', dest='display', help='Disables online display of tracker output (slow)',action='store_false')
    parser.add_argument('--dlib', dest='use_dlibTracker', help='Use dlib correlation tracker instead of kalman tracker',action='store_true')
    parser.add_argument('--save', dest='saver', help='Saves frames with tracking output, not used if --NoDisplay',action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


#get_bounding_box
# if __name__ == "__main__":
#
#     writer_open_pose = skvideo.io.FFmpegWriter(
#        "data/output.mp4",
#         inputdict = {
#             '-r': str(20)
#         },
#         outputdict = {
#             '-r': str(20)
#         }
#     )
#     writer_heat_map = skvideo.io.FFmpegWriter(
#         "data/heat_map.mp4",
#         inputdict={
#             '-r': str(20)
#         },
#         outputdict={
#             '-r': str(20)
#         }
#     )
#     video_capture = cv2.VideoCapture("output/TownCentreXVID.avi")
#     # fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # out_file = cv2.VideoWriter('',fourcc, 20.0, (360,480))
#     counter = 22
#     ret, frame = video_capture.read()
#     # while True:
#     heat_map = np.zeros((frame.shape[0], frame.shape[1],1), np.uint8)
#     # _ = handle_one(np.ones((540,960,3)), heat_map)
#
#
#     canvas, track_list = handle_one(frame)
#
#     while video_capture.isOpened():
#         counter-=1
#         start = time.clock()
#
#         # Capture frame-by-frame
#         ret, frame = video_capture.read()
#         print frame.shape
#
#         _, track_list = handle_one(frame)
#         for track_obj in track_list:
#             track_obj.visualize(frame)
#         # Display the resulting frame
#         # writer_open_pose.writeFrame(canvas)
#         # im_color = cv2.applyColorMap(heat_map, cv2.COLORMAP_HOT)
#         # cv2.imshow('Video', canvas)
#         # cv2.imshow('HeatMap', im_color)
#
#         # alpha = np.ones(im_color.shape, np.uint8)
#
#         # Normalize the alpha mask to keep intensity between 0 and 1
#         # alpha = alpha.astype(float) / 255
#
#         # Multiply the foreground with the alpha matte
#
#         # canvasbgr = cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
#         # for i in range(im_color.shape[0]):
#         #     for j in range(im_color.shape[1]):
#         #         alpha[i, j] = (im_color[i,j]+ canvas[i,j])/2
#                 # if (heat_map[i,j,0]==0):
#                 #     alpha[i,j] =canvas[i,j]
#                 # else:
#                 #     alpha[i,j] = im_color[i,j]
#
#         # writer_heat_map.writeFrame(cv2.cvtColor(alpha,cv2.COLOR_BGR2RGB))
#         # cv2.imshow("color", im_color)
#
#         cv2.imshow("canvas", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         print (1.0/(time.clock() - start))
#
#     # When everything is done, release the capture
#     writer_open_pose.close()
#     writer_heat_map.close()
#     video_capture.release()
#     cv2.destroyAllWindows()
