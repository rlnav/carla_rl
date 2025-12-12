#!/usr/bin/env python3

import os
import carla
import random
import time
import numpy as np
import cv2
from datetime import datetime

from carla_rl.src.config import IMG_HEIGHT, IMG_WIDTH

# Global variable to store latest frame (thread-safe for display)
latest_frame = None

def process_img(image, output_folder="captured_images"):
    global latest_frame
    # Convert CARLA raw data to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    array = array[:, :, :3]  # Drop alpha channel (BGR)
    
    # Store for main thread display
    latest_frame = array.copy()
    
    # Optional: save images
    # timestamp = int(time.time() * 1000)
    # cv2.imwrite(f"{output_folder}/carla_image_{timestamp}.png", array)

def main():
    global latest_frame
    
    # Connect to CARLA
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("Connected to CARLA server")
    except RuntimeError as e:
        print(f"Error: Failed to connect to CARLA server at localhost:2000")
        print(f"Make sure to run: ./CarlaUnreal.sh -RenderOffScreen -nosound")
        print(f"Details: {e}")
        return
    
    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
    except Exception as e:
        print(f"Error: Failed to get world or blueprints: {e}")
        return

    # Look for an existing vehicle
    try:
        actors = world.get_actors().filter("vehicle.*")
    except Exception as e:
        print(f"Error: Failed to get actors: {e}")
        return

    vehicle = None
    if actors:
        # Vehicle already exists → attach to the first one
        vehicle = actors[0]
        print(f"Connected to existing vehicle: id={vehicle.id}, type={vehicle.type_id}")
    else:
        # No vehicle found → spawn a new one
        try:
            spawn_point = random.choice(world.get_map().get_spawn_points())
            vehicle_bp = blueprint_library.find("vehicle.nissan.patrol")
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Spawned new vehicle: id={vehicle.id}, type={vehicle.type_id}")
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return

    # Create a camera
    try:
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        camera_bp.set_attribute("fov", "110")

        # Place camera in front of car
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("Camera created successfully")
    except Exception as e:
        print(f"Error creating camera: {e}")
        if vehicle:
            vehicle.destroy()
        return

    # Control the car
    try:
        throttle, steer = 1.0, 0.0
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        print(f"Controlling the car with throttle: {throttle} and steer: {steer}")
    except Exception as e:
        print(f"Warning: Could not control the car: {e}")

    # Start capturing images
    output_folder = f"outputs/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}"
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        camera.listen(lambda image: process_img(image, output_folder))
        print(f"Camera started capturing images (displaying in main thread)")
        print(f"Saving to: {output_folder}")
    except Exception as e:
        print(f"Error starting camera listener: {e}")
        camera.destroy()
        vehicle.destroy()
        return

    # Main display loop (handles both CARLA tick and OpenCV display)
    start_time = time.time()
    print("Press ESC to quit early...")
    try:
        while time.time() - start_time < 10:
            if latest_frame is not None:
                cv2.imshow("Car Camera", latest_frame)
            
            # 1ms waitKey for smooth display + ESC quit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("ESC pressed, quitting early")
                break
            
            time.sleep(0.01)  # Small sleep to prevent CPU spin
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        try:
            camera.stop()
            camera.destroy()
            vehicle.destroy()
        except:
            pass  # Ignore cleanup errors
        cv2.destroyAllWindows()
        print("Simulation finished.")

if __name__ == "__main__":
    main()
