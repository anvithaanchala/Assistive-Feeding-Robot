import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.interpolate import CubicSpline
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import modern_robotics as mr  # For IK solver

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the robot
bot = InterbotixManipulatorXS(robot_model='wx250s', group_name='arm', gripper_name='gripper')

def transform_coordinates(x, y, z):
    """
    Transforms image coordinates from the camera to the robot's base frame.
    """
    scale = 1  # Scale factor from normalized to meters
    x_offset = 0  # Example offsets
    y_offset = 0
    z_offset = 0.2  # Adjust to match robot's height
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    point_camera = np.array([x * scale, y * scale, z * scale])
    point_robot = np.dot(rotation_matrix, point_camera) + np.array([x_offset, y_offset, z_offset])
    return point_robot

def inverse_kinematics(target_position):
    """
    Solves for the joint angles using the Modern Robotics library.
    """
    # Robot’s home position transformation matrix
    M = np.array([
        [1, 0, 0, 0.4],  # End-effector home position (modify based on robot)
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])

    # Screw axis list for a 4-DOF arm (Modify as per robot's kinematics)
    S_list = np.array([
        [0, 0, 1, 0, 0, 0],    
        [0, 1, 0, -0.3, 0, 0], 
        [0, 1, 0, -0.3, 0, 0], 
        [0, 0, 1, 0, 0, 0.2],  
    ]).T

    initial_guess = np.zeros(4)  # Assuming a 4DOF arm
    target_transformation = np.eye(4)
    target_transformation[:3, 3] = target_position  # Assign desired (X, Y, Z)

    thetas, success = mr.IKinSpace(S_list, M, target_transformation, initial_guess, 0.01, 0.001)

    return thetas if success else None

def quintic_polynomial_interpolation(start, end, time_intervals):
    """
    Generates a smooth trajectory using Quintic Polynomial Interpolation.
    """
    t = np.linspace(0, 1, len(time_intervals))  # Normalize time
    cs = CubicSpline([0, 1], [start, end], bc_type=((1, 0.0), (1, 0.0)))  # Ensures smooth transition
    return cs(t)

def generate_cartesian_path(start, goal, num_points=10):
    """
    Generates waypoints between the food source and the user's mouth.
    """
    waypoints = np.linspace(start, goal, num=num_points)
    return waypoints

# Open video capture
cap = cv2.VideoCapture(0)
try:
    print("Press spacebar to capture and process an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue
        cv2.imshow("Real-time Video", frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):  # Press 'q' to quit
            break

        if key & 0xFF == ord(' '):  # Process the image when spacebar is pressed
            print("Processing image...")
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            print("Image processed.")

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    sum_x, sum_y, sum_z = 0, 0, 0
                    num_landmarks = 0
                    for idx in mp.solutions.face_mesh.FACEMESH_LIPS:
                        landmark1 = face_landmarks.landmark[idx[0]]
                        landmark2 = face_landmarks.landmark[idx[1]]
                        sum_x += landmark1.x + landmark2.x
                        sum_y += landmark1.y + landmark2.y
                        sum_z += landmark1.z + landmark2.z
                        num_landmarks += 2

                    if num_landmarks > 0:
                        centroid_x = sum_x / num_landmarks
                        centroid_y = sum_y / num_landmarks
                        centroid_z = sum_z / num_landmarks
                        print(f"Centroid in image coordinates: {centroid_x}, {centroid_y}, {centroid_z}")

                        # Transform coordinates to robot frame
                        x_robot, y_robot, z_robot = transform_coordinates(centroid_x, centroid_y, centroid_z)
                        print(f"Transformed coordinates: {x_robot}, {y_robot}, {z_robot}")

                        # Generate Cartesian path (Food source to Mouth)
                        food_source = np.array([0.3, -0.2, 0.1])  # Example food position
                        goal_position = np.array([x_robot, y_robot, z_robot])
                        waypoints = generate_cartesian_path(food_source, goal_position)

                        for point in waypoints:
                            joint_angles = inverse_kinematics(point)
                            if joint_angles is not None:
                                # Apply quintic polynomial interpolation for smooth movement
                                time_intervals = np.linspace(0, 1, len(waypoints))
                                smoothed_angles = quintic_polynomial_interpolation(bot.arm.get_joint_positions(), joint_angles, time_intervals)

                                for angles in smoothed_angles:
                                    bot.arm.set_joint_positions(angles)
                                    time.sleep(0.1)  # Allow the robot to reach position

                                print(f"Moved to joint angles: {joint_angles}")

                        # Draw a red circle at the centroid
                        cv2.circle(frame, (int(centroid_x * frame.shape[1]), int(centroid_y * frame.shape[0])), 5, (0, 0, 255), -1)
                        cv2.imshow("Real-time Video", frame)
                        print("Displayed centroid on video.")
            else:
                print("No face landmarks detected.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    bot.shutdown()
