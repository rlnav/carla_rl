#!/usr/bin/env python3

import os
import carla
import random
import time
import numpy as np
import cv2
from datetime import datetime


def process_img(image, output_folder="captured_images"):
    # Convert CARLA raw data to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    array = array[:, :, :3]  # Drop alpha channel (BGR)

    # save images
    cv2.imwrite(f"{output_folder}/carla_image_{int(time.time())}.png", array)


def main():
    # Connect to CARLA
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
    except RuntimeError as e:
        print(f"Error: Failed to connect to CARLA server at localhost:2000")
        print(f"Make sure to run: bash CarlaUnreal.sh -RenderOffScreen -nosound")
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
            # vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
            vehicle_bp = blueprint_library.find("vehicle.nissan.patrol")
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Spawned new vehicle: id={vehicle.id}, type={vehicle.type_id}")
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return

    # Create a camera
    try:
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        # Place camera in front of car
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("Camera created successfully")
    except Exception as e:
        print(f"Error creating camera: {e}")
        vehicle.destroy()
        return

    # Enable autopilot (CARLA 0.10 compatible)
    try:
        vehicle.set_autopilot(True)
        print("Autopilot enabled")
    except Exception as e:
        print(f"Warning: Could not enable autopilot: {e}")

    # Start capturing images
    output_folder = f"outputs/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}"
    os.makedirs(output_folder, exist_ok=True)
    try:
        camera.listen(lambda image: process_img(image, output_folder))
        print("Camera started capturing images")
    except Exception as e:
        print(f"Error starting camera listener: {e}")
        camera.destroy()
        vehicle.destroy()
        return

    # Keep running for 10 seconds
    try:
        print("Running simulation for 10 seconds...")
        time.sleep(10)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        try:
            camera.stop()
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        cv2.destroyAllWindows()
        print("Simulation finished.")


if __name__ == "__main__":
    main()