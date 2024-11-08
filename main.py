from djitellopy import tello
import threading
import keyboard
import time
import openai
import speech_recognition as sr
import os
from ultralytics import YOLO
from collections import Counter
import json
import cv2

SPEED = 50

# # Global variable to manage control state
# KEYBOARD_ACTIVE = False

def keyboard_control_drone():

    global KEYBOARD_ACTIVE
    print("Press space to takeoff...")
    keyboard.wait(hotkey="space")
    drone.takeoff()
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    # Loop to continuously check for keypresses
    try:
        while True:
            # Initialize velocity variables
            left_right_velocity = 0
            forward_backward_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
            KEYBOARD_ACTIVE = False

            # Forward/Backward
            if keyboard.is_pressed('w'):
                forward_backward_velocity = SPEED  # Move forward
                KEYBOARD_ACTIVE = True
            elif keyboard.is_pressed('s'):
                forward_backward_velocity = -SPEED  # Move backward
                KEYBOARD_ACTIVE = True

            # Left/Right (horizontal movement)
            if keyboard.is_pressed('a'):
                left_right_velocity = -SPEED  # Move left
                KEYBOARD_ACTIVE = True
            elif keyboard.is_pressed('d'):
                left_right_velocity = SPEED  # Move right
                KEYBOARD_ACTIVE = True

            # Up/Down (vertical movement)
            if keyboard.is_pressed('up'):
                up_down_velocity = SPEED  # Move up
                KEYBOARD_ACTIVE = True
            elif keyboard.is_pressed('down'):
                up_down_velocity = -SPEED  # Move down
                KEYBOARD_ACTIVE = True

            # Yaw control (rotation)
            if keyboard.is_pressed('left'):
                yaw_velocity = -SPEED  # Rotate counter-clockwise
                KEYBOARD_ACTIVE = True
            elif keyboard.is_pressed('right'):
                yaw_velocity = SPEED  # Rotate clockwise
                KEYBOARD_ACTIVE = True

            # Send the combined RC control commands to the drone
            if KEYBOARD_ACTIVE:
                drone.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
                # time.sleep(0.05)  # Small delay to allow the movement command to take effect
                drone.send_rc_control(0, 0, 0, 0)

            # Land the drone if spacebar is pressed
            if keyboard.is_pressed('space'):
                drone.land()
                stop_event.set()
                break



            time.sleep(0.05)  # Small delay to prevent CPU overload

    except:
        stop_event.set()
        drone.land()
        drone.end()

# def speech_control_drone():
#     client = openai.OpenAI()
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()

#     while True:
#         try:
#             with mic as source:
#                 recognizer.adjust_for_ambient_noise(source)
#                 print("Listening for voice commands...")
                
#                 audio = recognizer.listen(source)

#             with open("output.wav", "wb") as f:
#                 f.write(audio.get_wav_data())

            
#             # Send to OpenAI's Whisper API
#             audio_file = open("output.wav", "rb")
#             transcript = client.audio.transcriptions.create(
#                 file=audio_file,
#                 model="whisper-1",
#                 response_format="verbose_json",
#                 timestamp_granularities=["word"]
#             )

#             command = transcript.text.lower()

#             left_right_velocity = 0
#             forward_backward_velocity = 0
#             up_down_velocity = 0
#             yaw_velocity = 0

#             if "forward" in command:
#                 forward_backward_velocity = SPEED  # Move forward
#             elif "backward" in command:
#                 forward_backward_velocity = -SPEED # Move backward
            
#             if "left" in command:
#                 left_right_velocity = -SPEED  # Move left
#             elif "right" in command:
#                 left_right_velocity = SPEED  # Move right

#             if "up" in command:
#                 up_down_velocity = SPEED  # Move up
#             elif "down" in command:
#                 up_down_velocity = -SPEED  # Move down
                
#             if "rotate left" in command:
#                 yaw_velocity = -SPEED  # Rotate counter-clockwise
#             elif "rotate right" in command:
#                 yaw_velocity = SPEED  # Rotate clockwise

#             if "take off" in command:
#                 drone.takeoff()
#             elif "land" in command:
#                 drone.land()
#                 break
            
#             drone.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)

#         except:
#             drone.land()
#             print("An error ocurred in speech_control_drone()")
#             break

#         time.sleep(0.5)  # Short delay to avoid rapid re-triggering
        


def speech_to_text():
    client = openai.OpenAI()
    recognizer = sr.Recognizer()
    mic = sr.Microphone()


    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for voice commands...")
            
            audio = recognizer.listen(source)

            
        with open("output.wav", "wb") as f:
            f.write(audio.get_wav_data())

                
        # Send to OpenAI's Whisper API
        audio_file = open("output.wav", "rb")
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )

        command = transcript.text.lower()
        print(command)

        if command == "exit":
            break

def speech_commands():
    # Load the YOLOv8 model for object detection tasks
    image_model = YOLO('yolov8s.pt')

    # Set up OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    
    while True:
        try:
            with mic as source:
                print("Listening for launch or drone...")
                audio = recognizer.listen(source)

                
            with open("output.wav", "wb") as f:
                f.write(audio.get_wav_data())

                    
            # Send to OpenAI's Whisper API
            with open("output.wav", "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            command = transcript.text.lower()
            print(command)

            if "launch" in command:
                drone.takeoff()
                print(f'\033[34m{command}\033[0m')

            elif "drone" in command:
                print(f'\033[34m{command}\033[0m')
                question = ''
                while question == '':
                    with mic as source:
                        print("Listening for questions about what the drone sees...")
                        audio = recognizer.listen(source)

                    
                    with open("output.wav", "wb") as f:
                        f.write(audio.get_wav_data())

                        
                    # Send to OpenAI's Whisper API
                    with open("output.wav", "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            file=audio_file,
                            model="whisper-1",
                            response_format="verbose_json",
                            timestamp_granularities=["word"]
                        )

                    question = transcript.text.lower()
                    print(question)

                frame = drone.get_frame_read().frame
                # Save the frame as a JPEG image
                cv2.imwrite('frame.jpg', frame)

                # Send frame to YOLO model
                scene_description = object_detection("frame.jpg", image_model)

                # Combine them into a single prompt for clarity
                prompt = f"Scene description: {scene_description}\nQuestion: {question}"

                # Make the API call
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant that analyzes scene descriptions and answers questions based on them."},
                        {"role": "user", "content": prompt}
                    ]
                )
                print(f'\033[32m{response.choices[0].message.content}\n\033[0m')  # Print the response in green
        except KeyboardInterrupt:
            break
        except Exception as e:
            continue

        time.sleep(0.2)

def object_detection(image, model):
    '''
    Run object detection on an image frame using a YOLOv8 model from ultralytics library.
    Returns a string summary of the objects detected with their counts.
    '''
    # Run inference on an image
    results = model(image, verbose=False)
    detected_objects = results[0].to_json()

    # Parse JSON string into Python list
    detected_objects = json.loads(detected_objects)

    # Parse the results to get the count of each unique object it found
    object_counts = Counter(item["name"] for item in detected_objects)

    # Format object counts into a string to give to OpenAI API
    summary = ", ".join(f"{count} {name}" for name, count in object_counts.items())
    return summary

def stream_video(drone):
    '''
    https://github.com/Jacob-Pitsenberger/Tello-Flight-Routine-with-Video-Stream?tab=readme-ov-file
    '''
    while True:
        frame = drone.get_frame_read().frame
        # Check if frame reading was successful
        cv2.imshow('tello stream', frame)

        #######################  NEW - 3 ############################
        # Check if streaming readiness hasn't been signaled yet
        if not stream_ready.is_set():

            # Signal that video streaming is ready
            stream_ready.set()
            print("Event Signal Set: Stream is live.")
        ##########################################################

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__== "__main__":
    stream_ready = threading.Event()
    
    drone = tello.Tello()
    drone.connect()
    drone.streamon()

    # Create and start the streaming thread
    stream_thread = threading.Thread(target=stream_video, args=(drone,))

    # Set thread as a daemon thread to have it run in the background.
    # This allows our program to exit even if this streaming thread is still running after calling drone.reboot()
    # Also, this prevents errors in our video stream function from crashing our entire program if they occur.
    stream_thread.daemon = True

    # Start the thread
    stream_thread.start()

    print("drone connected and stream on\n")
    print(f'Battery Level: {drone.get_battery()}')

    proceed = input("Enter y/n to proceed:")

    if proceed == "y":
        speech_commands()
    else:
        drone.end()