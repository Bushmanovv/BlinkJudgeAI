import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist

class StaringContest:
    def __init__(self, num_players=2):
        # Validate number of players
        if num_players < 1 or num_players > 4:
            raise ValueError("Number of players must be between 1 and 4")

        self.num_players = num_players

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max(4, num_players),
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Correct eye landmarks indices for MediaPipe Face Mesh (468 landmarks)
        # These are the actual eye contour points
        self.LEFT_EYE = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        self.RIGHT_EYE = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]

        # Key points for EAR calculation (more reliable subset)
        self.LEFT_EYE_KEY = [33, 160, 158, 133, 153, 144]  # [outer, top1, top2, inner, bottom2, bottom1]
        self.RIGHT_EYE_KEY = [362, 385, 387, 263, 373, 380]  # [outer, top1, top2, inner, bottom2, bottom1]

        # Face boundary landmarks for head positioning
        self.FACE_BOUNDARY = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 125]

        # Blink detection parameters - adjusted for better accuracy
        self.EYE_AR_THRESH = 0.21  # Lowered threshold for more sensitive detection
        self.EYE_AR_CONSEC_FRAMES = 2  # Reduced frames needed for faster detection

        # Game state
        self.game_started = False
        self.game_over = False
        self.winner = None
        self.winners = []
        self.start_time = None

        # Player colors
        self.player_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (255, 255, 0)   # Cyan
        ]

        # Initialize players dynamically
        self.players = {}
        for i in range(num_players):
            self.players[i] = {
                'name': f'Player {i+1}',
                'blink_counter': 0,
                'consecutive_frames': 0,
                'lost': False,
                'color': self.player_colors[i],
                'detected': False,
                'ear_history': [],  # Track EAR history for better detection
                'baseline_ear': None,  # Individual baseline EAR
                'head_position': None  # Store head position for name display
            }

    def get_head_position(self, landmarks, img_width, img_height):
        """Get the top center of the head for name positioning"""
        if not landmarks or not landmarks.landmark:
            return None

        # Get forehead/top of head landmarks
        forehead_landmarks = [10, 151, 9, 8]  # Top center landmarks

        x_coords = []
        y_coords = []

        for idx in forehead_landmarks:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * img_width)
                y = int(landmarks.landmark[idx].y * img_height)
                x_coords.append(x)
                y_coords.append(y)

        if x_coords and y_coords:
            # Return center x and top y (with some offset above the head)
            center_x = sum(x_coords) // len(x_coords)
            top_y = min(y_coords) - 30  # 30 pixels above the head
            return (center_x, max(20, top_y))  # Ensure it's not too close to top of screen

        return None

    def draw_player_name(self, img, player_id, position):
        """Draw player name above their head with background"""
        if position is None:
            return

        player = self.players[player_id]
        name = player['name']
        color = player['color']

        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)

        # Calculate background rectangle
        x, y = position
        bg_x1 = x - text_width // 2 - 5
        bg_y1 = y - text_height - 5
        bg_x2 = x + text_width // 2 + 5
        bg_y2 = y + 5

        # Draw background rectangle
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

        # Draw player name
        text_x = x - text_width // 2
        text_y = y - 5
        cv2.putText(img, name, (text_x, text_y), font, font_scale, color, thickness)

        # Add status indicator
        if player['lost']:
            status = "LOST"
            status_color = (0, 0, 255)  # Red
        elif not player['detected']:
            status = "NOT DETECTED"
            status_color = (128, 128, 128)  # Gray
        else:
            status = "ACTIVE"
            status_color = (0, 255, 0)  # Green

        # Draw status below name
        status_font_scale = 0.4
        (status_width, status_height), _ = cv2.getTextSize(status, font, status_font_scale, 1)
        status_x = x - status_width // 2
        status_y = y + 15
        cv2.putText(img, status, (status_x, status_y), font, status_font_scale, status_color, 1)

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate the eye aspect ratio (EAR) for blink detection"""
        # Ensure we have exactly 6 points [outer, top1, top2, inner, bottom2, bottom1]
        if len(eye_landmarks) != 6:
            return 0.0

        # Compute the euclidean distances between the vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])  # top1 to bottom1
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])  # top2 to bottom2

        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  # outer to inner

        # Compute the eye aspect ratio
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return ear

    def extract_eye_landmarks(self, landmarks, eye_indices, img_width, img_height):
        """Extract eye landmarks coordinates"""
        coords = []
        for idx in eye_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * img_width)
                y = int(landmarks.landmark[idx].y * img_height)
                coords.append((x, y))
        return coords

    def detect_blink(self, landmarks, player_id, img_width, img_height):
        """Detect if a player blinked using improved method"""
        # Extract key eye coordinates for EAR calculation
        left_eye_coords = self.extract_eye_landmarks(landmarks, self.LEFT_EYE_KEY, img_width, img_height)
        right_eye_coords = self.extract_eye_landmarks(landmarks, self.RIGHT_EYE_KEY, img_width, img_height)

        # Extract full eye contours for visualization
        left_eye_full = self.extract_eye_landmarks(landmarks, self.LEFT_EYE, img_width, img_height)
        right_eye_full = self.extract_eye_landmarks(landmarks, self.RIGHT_EYE, img_width, img_height)

        # Get head position for name display
        head_pos = self.get_head_position(landmarks, img_width, img_height)
        self.players[player_id]['head_position'] = head_pos

        if len(left_eye_coords) != 6 or len(right_eye_coords) != 6:
            return 0.0, left_eye_full, right_eye_full

        # Calculate EAR for both eyes
        left_ear = self.calculate_eye_aspect_ratio(left_eye_coords)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_coords)
        ear = (left_ear + right_ear) / 2.0

        player = self.players[player_id]
        player['detected'] = True

        # Update EAR history for baseline calculation
        player['ear_history'].append(ear)
        if len(player['ear_history']) > 10:  # Keep last 10 values
            player['ear_history'].pop(0)

        # Calculate adaptive baseline EAR (average of recent values when eyes are open)
        if len(player['ear_history']) >= 5:
            # Use upper quartile as baseline (when eyes are more open)
            sorted_ears = sorted(player['ear_history'])
            player['baseline_ear'] = sorted_ears[int(len(sorted_ears) * 0.75)]

            # Use adaptive threshold based on individual baseline
            adaptive_thresh = player['baseline_ear'] * 0.75  # 75% of baseline
            thresh = max(adaptive_thresh, 0.15)  # Minimum threshold
        else:
            thresh = self.EYE_AR_THRESH

        # Check if EAR is below threshold (eyes closed)
        if ear < thresh:
            player['consecutive_frames'] += 1

            # If eyes closed for enough consecutive frames, it's a blink
            if player['consecutive_frames'] >= self.EYE_AR_CONSEC_FRAMES:
                if player['consecutive_frames'] == self.EYE_AR_CONSEC_FRAMES:  # Only count once
                    player['blink_counter'] += 1
                    print(f"{player['name']} blinked! (EAR: {ear:.3f}, Threshold: {thresh:.3f})")

                if self.game_started and not self.game_over and not player['lost']:
                    player['lost'] = True
                    print(f"{player['name']} is out!")
                    self.check_game_end()
        else:
            player['consecutive_frames'] = 0

        return ear, left_eye_full, right_eye_full

    def draw_eye_landmarks(self, img, eye_coords, color):
        """Draw eye landmarks on the image"""
        if len(eye_coords) > 0:
            # Draw eye contour
            points = np.array(eye_coords, dtype=np.int32)
            cv2.polylines(img, [points], True, color, 1)

            # Draw key points
            for point in eye_coords:
                cv2.circle(img, point, 1, color, -1)

    def draw_game_info(self, img, detected_faces):
        """Draw game information on the image"""
        height, width = img.shape[:2]

        # Draw background for text
        info_height = 140 + (self.num_players * 30)
        cv2.rectangle(img, (10, 10), (width - 10, info_height), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (width - 10, info_height), (255, 255, 255), 2)

        if not self.game_started:
            cv2.putText(img, f"STARING CONTEST ({self.num_players} PLAYERS)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Press SPACE to start", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, "First person to blink loses!", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, f"Detected faces: {detected_faces}/{self.num_players}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(img, "Calibrating blink detection...", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(img, "Press 'q' to quit", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif self.game_over:
            if len(self.winners) == 1:
                winner_name = self.players[self.winners[0]]['name']
                cv2.putText(img, f"{winner_name} WINS!", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif len(self.winners) > 1:
                winner_names = [self.players[w]['name'] for w in self.winners]
                cv2.putText(img, f"TIE: {', '.join(winner_names)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(img, "Everyone blinked!", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.putText(img, "Press SPACE to play again", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.start_time:
                duration = time.time() - self.start_time
                cv2.putText(img, f"Duration: {duration:.1f}s", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(img, "GAME IN PROGRESS - DON'T BLINK!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if self.start_time:
                duration = time.time() - self.start_time
                cv2.putText(img, f"Time: {duration:.1f}s", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            active_players = sum(1 for p in self.players.values() if not p['lost'])
            cv2.putText(img, f"Active: {active_players}/{self.num_players}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw player info with EAR values
        y_start = height - 35 - (self.num_players * 30)
        for i, (player_id, player) in enumerate(self.players.items()):
            color = player['color']
            if player['lost']:
                status = "LOST"
                color = (128, 128, 128)
            elif not player['detected']:
                status = "NOT DETECTED"
                color = (0, 0, 255)
            else:
                status = "ACTIVE"

            # Show baseline EAR if available
            baseline_text = ""
            if player['baseline_ear'] is not None:
                baseline_text = f" (Baseline: {player['baseline_ear']:.3f})"

            text = f"{player['name']}: {status} - Blinks: {player['blink_counter']}{baseline_text}"
            cv2.putText(img, text, (20, y_start + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def check_game_end(self):
        """Check if game should end and determine winners"""
        if not self.game_started or self.game_over:
            return

        active_players = [pid for pid, player in self.players.items() if not player['lost']]

        if self.num_players == 1:
            if self.players[0]['lost']:
                self.game_over = True
                self.winners = []
        else:
            if len(active_players) <= 1:
                self.game_over = True
                self.winners = active_players

    def reset_game(self):
        """Reset the game state"""
        self.game_started = False
        self.game_over = False
        self.winner = None
        self.winners = []
        self.start_time = None

        for player in self.players.values():
            player['blink_counter'] = 0
            player['consecutive_frames'] = 0
            player['lost'] = False
            player['detected'] = False
            player['head_position'] = None
            # Keep EAR history for consistent baseline

    def set_player_names(self, names):
        """Set custom names for players"""
        for i, name in enumerate(names):
            if i < self.num_players:
                self.players[i]['name'] = name

    def list_available_cameras(self):
        """List all available cameras with their details"""
        print("\n🎥 Scanning for available cameras...")
        available_cameras = []

        camera_names = {
            0: "MacBook Built-in Camera",
            1: "iPhone/iPad Continuity Camera",
            2: "External USB Camera",
            3: "Additional Camera Device",
            4: "Additional Camera Device"
        }

        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    camera_info = {
                        'index': i,
                        'name': camera_names.get(i, f"Camera {i}"),
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    }
                    available_cameras.append(camera_info)
                    print(f"✅ Camera {i}: {camera_info['name']} - {camera_info['resolution']} @ {fps:.0f}fps")
                else:
                    print(f"⚠️  Camera {i}: {camera_names.get(i, f'Camera {i}')} - Opens but no frames")
                cap.release()

        if not available_cameras:
            print("❌ No working cameras found!")
            print("\n💡 Troubleshooting tips:")
            print("   • Make sure no other apps are using cameras")
            print("   • Check camera permissions in system settings")
        else:
            print(f"\n✅ Found {len(available_cameras)} working camera(s)")

        return available_cameras

    def run(self, camera_index=None, show_camera_list=True):
        """Main game loop"""
        if show_camera_list:
            available_cameras = self.list_available_cameras()
            if not available_cameras:
                return

            if camera_index is None:
                print(f"\n🎯 Auto-selecting Camera {available_cameras[0]['index']}: {available_cameras[0]['name']}")
                camera_index = available_cameras[0]['index']

        cap = cv2.VideoCapture(camera_index if camera_index is not None else 0)

        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"🚀 Starting game with Camera {camera_index} ({width}x{height} @ {fps:.0f}fps)")

        print("\nStaring Contest Started!")
        print("Instructions:")
        print(f"- Position {self.num_players} people in front of the camera")
        print("- Player names will appear above heads")
        print("- Wait for blink detection to calibrate (few seconds)")
        print("- Press SPACE to start the contest")
        print("- First person to blink loses!")
        print("- Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]

            # Reset player detection status
            for player in self.players.values():
                player['detected'] = False

            # Process frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)

            detected_faces = 0
            if results.multi_face_landmarks:
                detected_faces = len(results.multi_face_landmarks)

                # Process up to num_players faces
                for i, face_landmarks in enumerate(results.multi_face_landmarks[:self.num_players]):
                    if i < self.num_players:
                        ear, left_eye, right_eye = self.detect_blink(face_landmarks, i, width, height)

                        # Draw eye landmarks
                        color = self.players[i]['color']
                        self.draw_eye_landmarks(frame, left_eye, color)
                        self.draw_eye_landmarks(frame, right_eye, color)

                        # Draw player name above head
                        head_pos = self.players[i]['head_position']
                        self.draw_player_name(frame, i, head_pos)

                        # Draw EAR value and threshold
                        player = self.players[i]
                        thresh = player['baseline_ear'] * 0.75 if player['baseline_ear'] else self.EYE_AR_THRESH
                        thresh = max(thresh, 0.15)

                        ear_color = (0, 255, 0) if ear >= thresh else (0, 0, 255)
                        cv2.putText(frame, f"P{i+1}: {ear:.3f} (T:{thresh:.3f})",
                                    (20 + i * 200, height - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ear_color, 1)

                # Show warning if not enough players detected
                if detected_faces < self.num_players:
                    cv2.putText(frame, f"Need {self.num_players} players! Currently: {detected_faces}",
                                (20, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw game information
            self.draw_game_info(frame, detected_faces)

            # Show frame
            cv2.imshow('Staring Contest - With Player Names', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not self.game_started or self.game_over:
                    detected_count = sum(1 for p in self.players.values() if p['detected'])
                    # Check if all players have baseline calibrated
                    calibrated_count = sum(1 for p in self.players.values()
                                           if p['baseline_ear'] is not None)

                    if detected_count >= self.num_players and calibrated_count >= self.num_players:
                        self.reset_game()
                        self.game_started = True
                        self.start_time = time.time()
                        print(f"Game started with {self.num_players} players! Don't blink!")
                    else:
                        print(f"Wait for calibration! Detected: {detected_count}/{self.num_players}, Calibrated: {calibrated_count}/{self.num_players}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from scipy.spatial import distance as dist
    except ImportError as e:
        print("Missing required package:", str(e))
        print("Please install required packages:")
        print("pip install opencv-python mediapipe numpy scipy")
        exit(1)

    print("🎮 STARING CONTEST - With Player Names Above Heads")
    print("=" * 60)

    # Get number of players
    while True:
        try:
            num_players = int(input("Enter number of players (1-4): "))
            if 1 <= num_players <= 4:
                break
            else:
                print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")

    game = StaringContest(num_players=num_players)

    # Optional: Get custom player names
    custom_names = input(f"\nEnter custom names for {num_players} players (comma-separated), or press Enter for default: ").strip()
    if custom_names:
        names = [name.strip() for name in custom_names.split(',')]
        if len(names) >= num_players:
            game.set_player_names(names[:num_players])
            print(f"Using custom names: {', '.join(names[:num_players])}")
        else:
            print("Not enough names provided, using default names")

    print(f"\n🎯 Game Mode: {num_players} Player{'s' if num_players > 1 else ''}")
    print("\n🔧 New Features:")
    print("• Player names displayed above heads")
    print("• Real-time status indicators")
    print("• Improved visual feedback")
    print("• Adaptive blink detection")

    game.run()