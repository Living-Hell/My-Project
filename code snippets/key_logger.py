import csv
import time
from collections import defaultdict
from pynput import keyboard

# Variables to store key press details
key_count = 0
key_press_durations = {}
time_between_keys = []
last_key_time = None
start_time = time.time()
key_sequences = []
backspace_count = 0

# File to store the data
csv_file = 'key_press_data.csv'

# User identification
user_id = 'User123'  # Replace with actual user identification mechanism

def on_press(key):
    global last_key_time, key_count, backspace_count

    # Record key down time
    current_time = time.time()
    
    # Calculate time between key presses
    if last_key_time is not None:
        time_between_keys.append(current_time - last_key_time)
    last_key_time = current_time

    # Record key sequence
    key_sequences.append(str(key))
    
    # Increase the key press count
    key_count += 1
    
    # Track backspace usage
    if key == keyboard.Key.backspace:
        backspace_count += 1

def on_release(key):
    global start_time, key_count
    
    # Calculate duration of the key press
    duration = time.time() - last_key_time
    
    # Calculate typing speed (keys per second)
    elapsed_time = time.time() - start_time
    typing_speed = key_count / elapsed_time
    
    # Calculate time between keys
    if len(time_between_keys) > 0:
        time_between = time_between_keys[-1]
    else:
        time_between = None

    # Write data to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            user_id,  # User ID
            str(key), 
            duration, 
            time_between, 
            typing_speed, 
            backspace_count
        ])

    # Exit the listener if 'esc' key is pressed
    if key == keyboard.Key.esc:
        return False

def main():
    # Create or clear the CSV file and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'User ID',  # Add user ID as a header
            'Key', 
            'Duration', 
            'Time Between Keys', 
            'Typing Speed (KPS)', 
            'Backspace Count'
        ])

    # Start the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    main()
