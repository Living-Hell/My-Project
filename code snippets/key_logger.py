import csv
import time
from collections import Counter
from pynput import keyboard

# Variables to store key press details
key_count = 0
key_press_durations = {}
time_between_keys = []
last_key_time = None
start_time = time.time()
key_sequences = []
backspace_count = 0
idle_times = []
idle_start_time = None
key_frequency = Counter()

# Files to store the data
csv_file = 'key_press_data.csv'
summary_csv_file = 'summary_statistics.csv'

# Prompt for user ID
user_id = input("Please enter your user ID: ")

def on_press(key):
    global last_key_time, key_count, backspace_count, idle_start_time

    # Record current time
    current_time = time.time()

    # Calculate idle time
    if last_key_time is not None:
        idle_time = current_time - last_key_time
        idle_times.append(idle_time)
    
    # Reset idle start time
    idle_start_time = current_time
    
    # Record key sequence
    key_sequences.append(key)
    
    # Increase the key press count
    key_count += 1
    
    # Track key frequency
    key_frequency[key] += 1
    
    # Track backspace usage
    if key == keyboard.Key.backspace:
        backspace_count += 1

def on_release(key):
    global start_time, key_count, last_key_time

    # Record key up time
    current_time = time.time()
    
    # Calculate duration of the key press
    duration = current_time - last_key_time if last_key_time else 0
    key_press_durations[key] = duration
    
    # Calculate typing speed (keys per second)
    elapsed_time = time.time() - start_time
    typing_speed = key_count / elapsed_time if elapsed_time > 0 else 0
    
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
            backspace_count,
            elapsed_time
        ])

    # Update last key time
    last_key_time = current_time

    # Exit the listener if 'esc' key is pressed
    if key == keyboard.Key.esc:
        return False

def save_summary_statistics():
    # Calculate summary statistics
    elapsed_time = time.time() - start_time
    avg_idle_time = sum(idle_times) / len(idle_times) if idle_times else 0
    most_frequent_key = key_frequency.most_common(1)[0][0] if key_frequency else None
    least_frequent_key = key_frequency.most_common()[-1][0] if key_frequency else None
    typing_speed_variability = max(idle_times) - min(idle_times) if idle_times else 0
    avg_key_press_intensity = sum(key_press_durations.values()) / len(key_press_durations) if key_press_durations else 0
    error_rate = backspace_count / key_count if key_count > 0 else 0
    shift_usage = key_frequency[keyboard.Key.shift] if keyboard.Key.shift in key_frequency else 0
    special_char_usage = sum(key_frequency[key] for key in key_frequency if hasattr(key, 'char') and not key.char.isalnum())
    
    # Write summary statistics to a separate CSV file
    with open(summary_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['User ID', user_id])  # Include user ID in summary
        writer.writerow(['Total Keys Pressed', key_count])
        writer.writerow(['Typing Session Duration (s)', elapsed_time])
        writer.writerow(['Average Idle Time (s)', avg_idle_time])
        writer.writerow(['Most Frequently Pressed Key', most_frequent_key])
        writer.writerow(['Least Frequently Pressed Key', least_frequent_key])
        writer.writerow(['Typing Speed Variability (s)', typing_speed_variability])
        writer.writerow(['Average Key Press Intensity (s)', avg_key_press_intensity])
        writer.writerow(['Error Rate', error_rate])
        writer.writerow(['Shift Key Usage', shift_usage])
        writer.writerow(['Special Character Usage', special_char_usage])

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
            'Backspace Count',
            'Typing Session Duration'
        ])

    # Start the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    
    # Save summary statistics
    save_summary_statistics()

if __name__ == '__main__':
    main()
