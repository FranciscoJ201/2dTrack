from moviepy import VideoFileClip, clips_array

def create_side_by_side(input1_path, input2_path, output_path):
    """
    Takes two video files and combines them side-by-side into a single video.
    """
    
    # Load the video clips
    clip1 = VideoFileClip(input1_path)
    clip2 = VideoFileClip(input2_path)
    
    # Resize clip2 to match clip1's height to prevent clips_array from crashing
    if clip1.h != clip2.h:
        clip2 = clip2.resized(height=clip1.h)
        
    # Arrange the clips horizontally in a 1x2 grid
    final_clip = clips_array([[clip1, clip2]])
    
    # Write the result to a file (using standard MP4 codecs)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Close the clips to free up system memory
    clip1.close()
    clip2.close()
    final_clip.close()

if __name__ == "__main__":
    # Example usage:
    # Set these to your actual file names
    left_video = '/Users/franciscojimenez/Desktop/sumovid.mp4'
    right_video = 'sumovid_plot_output.mp4'
    final_output = "side_by_side_comparison.mp4"
    
    create_side_by_side(left_video, right_video, final_output)