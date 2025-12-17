import cv2
import random
import mediapipe as mp
import numpy as np
from fruit_object import Fruit
from particle_effect import ParticleSystem, SlicedFruit
from collections import deque

class FruitSliceGame:
    def __init__(self):
        # Window settings - Fullscreen
        self.window_name = "Fruit Slice Game - Hand Gesture"
        
        # Create fullscreen window first to get dimensions
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Get actual screen resolution
        import ctypes
        user32 = ctypes.windll.user32
        self.window_width = user32.GetSystemMetrics(0)
        self.window_height = user32.GetSystemMetrics(1)
        
        # Camera settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Request high res
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # MediaPipe Hand Tracking - HIGH ACCURACY MODE
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,   # 90% akurasi deteksi 
            min_tracking_confidence=0.8     # 80% akurasi tracking 
        )
        
        # Hand trail for slicing effect
        self.hand_trail = deque(maxlen=20)  # Store last 20 positions
        
        # Particle system for effects
        self.particle_system = ParticleSystem()
        self.sliced_fruits = []  # Store sliced fruit halves
        
        # Game objects
        self.fruits = []
        self.max_fruits = 6  # Jumlah buah dikurangi
        self.fruit_spawn_counter = 0
        self.fruit_spawn_rate = 20  # Spawn lebih lambat
        
        # Game state
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.max_combo = 0
        self.game_over = False
        self.game_started = False  # New: Menu start
        self.running = True
        
        # Button states for game over
        self.button_restart = None
        self.button_quit = None
        self.button_start = None  # New: Start button
        self.mouse_x = 0
        self.mouse_y = 0
        
    def spawn_fruit(self):
        """Spawn a new fruit at random position"""
        if len(self.fruits) < self.max_fruits:
            fruit = Fruit(self.window_width, self.window_height)
            self.fruits.append(fruit)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for buttons"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start screen button
            if not self.game_started and self.button_start:
                bx, by, bw, bh = self.button_start
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.game_started = True
                    return
            
            # Game over buttons
            if self.game_over:
                # Check if restart button clicked
                if self.button_restart:
                    bx, by, bw, bh = self.button_restart
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.reset_game()
                
                # Check if quit button clicked
                if self.button_quit:
                    bx, by, bw, bh = self.button_quit
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.running = False
    
    def update_fruits(self):
        """Update all fruits position"""
        # Remove fruits that are out of screen (lose life if not sliced)
        fruits_to_remove = []
        for fruit in self.fruits:
            if fruit.is_out_of_screen():
                if not fruit.is_sliced and not self.game_over:
                    self.lives -= 1
                    self.combo = 0  # Reset combo
                    print(f"Missed fruit! Lives remaining: {self.lives}")
                    if self.lives <= 0:
                        self.game_over = True
                        print("Game Over!")
                fruits_to_remove.append(fruit)
        
        # Remove fruits
        for fruit in fruits_to_remove:
            self.fruits.remove(fruit)
        
        # Update remaining fruits
        for fruit in self.fruits:
            fruit.update()
        
        # Update sliced fruits animation
        for sliced in self.sliced_fruits:
            sliced.update()
        
        # Remove dead sliced fruits
        self.sliced_fruits = [s for s in self.sliced_fruits if not s.is_dead()]
        
        # Update particle system
        self.particle_system.update()
    
    def draw_fruits(self, frame):
        """Draw all fruits on frame"""
        # Draw sliced fruit halves first (behind)
        for sliced in self.sliced_fruits:
            sliced.draw(frame)
        
        # Draw particles
        self.particle_system.draw(frame)
        
        # Draw active fruits
        for fruit in self.fruits:
            if not fruit.is_sliced:
                fruit.draw(frame)
    
    def process_hand_tracking(self, frame):
        """Process hand tracking and detect slicing"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and check collision
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                
                # ============ VALIDASI JARI TELUNJUK LURUS (EXTENDED) ============
                # Ambil landmark jari telunjuk (5=MCP, 6=PIP, 7=DIP, 8=TIP)
                index_mcp = hand_landmarks.landmark[5]   # Base jari telunjuk
                index_pip = hand_landmarks.landmark[6]   # Joint tengah
                index_dip = hand_landmarks.landmark[7]   # Joint atas
                index_tip = hand_landmarks.landmark[8]   # Ujung jari
                
                # VALIDASI 1: Jari harus extended (TIP jauh lebih tinggi dari MCP)
                # Y coordinates: semakin kecil = semakin atas
                vertical_distance = index_mcp.y - index_tip.y
                finger_is_up = vertical_distance > 0.15  # Minimal 15% dari frame height
                
                # VALIDASI 2: Semua joint harus terurut dari bawah ke atas
                joints_in_order = (
                    index_mcp.y > index_pip.y > index_dip.y > index_tip.y
                )
                
                # VALIDASI 3: Hitung panjang jari (jarak MCP ke TIP)
                import math
                finger_length = math.sqrt(
                    (index_tip.x - index_mcp.x)**2 + 
                    (index_tip.y - index_mcp.y)**2
                )
                
                # Jari harus cukup panjang (tidak dilipat)
                finger_is_extended = finger_length > 0.12  # Minimal 12% diagonal
                
                # VALIDASI 4: JARI TELUNJUK HARUS PALING TINGGI (bukan jari lain)
                # Cek jari tengah, manis, kelingking HARUS lebih rendah dari telunjuk
                middle_tip = hand_landmarks.landmark[12]   # Jari tengah
                ring_tip = hand_landmarks.landmark[16]     # Jari manis  
                pinky_tip = hand_landmarks.landmark[20]    # Kelingking
                thumb_tip = hand_landmarks.landmark[4]     # Jempol
                
                # Jari telunjuk HARUS paling tinggi (y paling kecil)
                index_is_highest = (
                    index_tip.y < middle_tip.y - 0.03 and  # Telunjuk > tengah
                    index_tip.y < ring_tip.y - 0.03 and    # Telunjuk > manis
                    index_tip.y < pinky_tip.y - 0.03       # Telunjuk > kelingking
                )
                
                # GABUNGAN: Semua validasi harus TRUE
                is_valid_gesture = (
                    finger_is_up and 
                    joints_in_order and 
                    finger_is_extended and
                    index_is_highest  # ← PENTING: Telunjuk harus tertinggi!
                )
                
                # HANYA proses jika gesture VALID
                if is_valid_gesture:
                    x = int(index_tip.x * self.window_width)
                    y = int(index_tip.y * self.window_height)
                    
                    # Add to trail
                    self.hand_trail.append((x, y))
                else:
                    # Gesture tidak valid = CLEAR TRAIL IMMEDIATELY
                    if len(self.hand_trail) > 0:
                        self.hand_trail.clear()
                
                # Check collision HANYA jika gesture valid
                if len(self.hand_trail) >= 2 and is_valid_gesture:
                    p1 = self.hand_trail[-2]
                    p2 = self.hand_trail[-1]
                    
                    for fruit in self.fruits:
                        if fruit.check_line_collision(p1, p2):
                            fruit.slice()
                            self.score += 10
                            self.combo += 1
                            self.max_combo = max(self.max_combo, self.combo)
                            
                            # Create particle effect
                            fruit_colors = {
                                'straw': (0, 0, 255),
                                'orange': (0, 165, 255),
                                'banana': (0, 255, 255),
                                'nanas': (0, 200, 255),
                                'watermelon': (0, 255, 0)
                            }
                            color = fruit_colors.get(fruit.fruit_type, (255, 255, 255))
                            self.particle_system.emit(int(fruit.x), int(fruit.y), color, count=30)
                            
                            # Create sliced fruit animation
                            sliced = SlicedFruit(fruit)
                            self.sliced_fruits.append(sliced)
        else:
            # Reset combo if no hand detected for too long
            if len(self.hand_trail) > 0:
                self.hand_trail.clear()
        
        # Draw hand trail (slicing line)
        if len(self.hand_trail) > 1:
            for i in range(1, len(self.hand_trail)):
                thickness = int((i / len(self.hand_trail)) * 5) + 1
                cv2.line(frame, self.hand_trail[i-1], self.hand_trail[i], 
                        (0, 255, 255), thickness)
    
    def draw_ui(self, frame):
        """Draw modern UI - Score dan Nyawa di bawah kiri"""
        # Calculate responsive sizes
        title_font = self.window_height / 600.0
        ui_font = self.window_height / 900.0
        padding = int(self.window_height / 40)
        
        # Modern Title - Center Top dengan glow effect
        title = "FRUIT SLICE"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_font * 1.5, 4)[0]
        title_x = (self.window_width - title_size[0]) // 2
        # Glow effect
        cv2.putText(frame, title, (title_x, 80), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font * 1.5, (255, 150, 0), 10, cv2.LINE_AA)
        cv2.putText(frame, title, (title_x, 80), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font * 1.5, (0, 255, 255), 5, cv2.LINE_AA)
        
        # === BOTTOM LEFT - Score & Lives Cards ===
        card_w, card_h = 300, 120
        bottom_margin = padding + 20
        
        # Score Card - Bottom Left
        score_x = padding
        score_y = self.window_height - bottom_margin - card_h
        self._draw_card(frame, score_x, score_y, card_w, card_h, (40, 40, 40))
        
        cv2.putText(frame, "SCORE", (score_x + 20, score_y + 40), 
                   cv2.FONT_HERSHEY_DUPLEX, ui_font * 0.8, (150, 150, 150), 2)
        cv2.putText(frame, str(self.score), (score_x + 20, score_y + 95), 
                   cv2.FONT_HERSHEY_DUPLEX, ui_font * 2.2, (0, 255, 100), 4)
        
        # Lives Card - Above Score
        lives_x = padding
        lives_y = score_y - card_h - 20
        self._draw_card(frame, lives_x, lives_y, card_w, card_h, (40, 40, 40))
        
        cv2.putText(frame, "NYAWA", (lives_x + 20, lives_y + 40), 
                   cv2.FONT_HERSHEY_DUPLEX, ui_font * 0.8, (150, 150, 150), 2)
        
        # Draw hearts untuk lives
        heart_y = lives_y + 80
        heart_spacing = 70
        for i in range(3):
            heart_x = lives_x + 40 + (i * heart_spacing)
            if i < self.lives:
                # Red heart (alive)
                cv2.circle(frame, (heart_x - 15, heart_y - 10), 14, (0, 0, 255), -1)
                cv2.circle(frame, (heart_x + 15, heart_y - 10), 14, (0, 0, 255), -1)
                pts = np.array([[heart_x - 28, heart_y - 5], [heart_x, heart_y + 22], 
                               [heart_x + 28, heart_y - 5]], np.int32)
                cv2.fillPoly(frame, [pts], (0, 0, 255))
            else:
                # Gray heart (lost)
                cv2.circle(frame, (heart_x - 15, heart_y - 10), 14, (80, 80, 80), -1)
                cv2.circle(frame, (heart_x + 15, heart_y - 10), 14, (80, 80, 80), -1)
                pts = np.array([[heart_x - 28, heart_y - 5], [heart_x, heart_y + 22], 
                               [heart_x + 28, heart_y - 5]], np.int32)
                cv2.fillPoly(frame, [pts], (80, 80, 80))
        
        # Combo display - Center dengan glow (hanya saat aktif)
        if self.combo > 1:
            combo_text = f"COMBO x{self.combo}!"
            combo_font = self.window_height / 350.0
            text_size = cv2.getTextSize(combo_text, cv2.FONT_HERSHEY_DUPLEX, combo_font, 5)[0]
            text_x = (self.window_width - text_size[0]) // 2
            text_y = self.window_height // 3
            
            # Animated glow effect
            import math
            pulse = abs(math.sin(cv2.getTickCount() / 1000.0)) * 0.3 + 0.7
            glow_color = (0, int(255 * pulse), int(255 * pulse))
            
            # Outer glow
            cv2.putText(frame, combo_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 
                       combo_font, glow_color, 10, cv2.LINE_AA)
            # Inner text
            cv2.putText(frame, combo_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 
                       combo_font, (255, 255, 0), 5, cv2.LINE_AA)
        
        # Right corner - Active fruits indicator
        cv2.putText(frame, f"Fruits: {len(self.fruits)}", 
                   (self.window_width - 230, 80), cv2.FONT_HERSHEY_DUPLEX, 
                   ui_font * 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Modern Game Over Screen dengan 2 Button
        if self.game_over:
            # Full dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
            
            # Center card - lebih besar
            card_w, card_h = 900, 650
            card_x = (self.window_width - card_w) // 2
            card_y = (self.window_height - card_h) // 2
            self._draw_card(frame, card_x, card_y, card_w, card_h, (30, 30, 30), border_color=(255, 50, 50))
            
            # Game Over text dengan glow - CENTERED
            gameover_font = self.window_height / 250.0
            game_over_text = "GAME OVER"
            text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_DUPLEX, gameover_font, 6)[0]
            text_x = (self.window_width - text_size[0]) // 2
            text_y = card_y + 120
            
            cv2.putText(frame, game_over_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 
                       gameover_font, (100, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(frame, game_over_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 
                       gameover_font, (255, 50, 50), 6, cv2.LINE_AA)
            
            # Stats in 2 columns - CENTERED dan RAPI
            score_font = self.window_height / 500.0
            stats_y = text_y + 140
            
            # Calculate center positions for 2 columns
            col_spacing = 350
            col1_x = (self.window_width - col_spacing) // 2 - 100
            col2_x = (self.window_width + col_spacing) // 2 - 100
            
            # SKOR AKHIR - Left Column
            label1 = "SKOR AKHIR"
            label1_size = cv2.getTextSize(label1, cv2.FONT_HERSHEY_DUPLEX, score_font * 0.9, 2)[0]
            label1_x = col1_x + (200 - label1_size[0]) // 2
            cv2.putText(frame, label1, (label1_x, stats_y), 
                       cv2.FONT_HERSHEY_DUPLEX, score_font * 0.9, (150, 150, 150), 2)
            
            score_text = str(self.score)
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, score_font * 2.5, 5)[0]
            score_x = col1_x + (200 - score_size[0]) // 2
            cv2.putText(frame, score_text, (score_x, stats_y + 120), 
                       cv2.FONT_HERSHEY_DUPLEX, score_font * 2.5, (0, 255, 100), 5)
            
            # COMBO MAX - Right Column
            label2 = "COMBO MAX"
            label2_size = cv2.getTextSize(label2, cv2.FONT_HERSHEY_DUPLEX, score_font * 0.9, 2)[0]
            label2_x = col2_x + (200 - label2_size[0]) // 2
            cv2.putText(frame, label2, (label2_x, stats_y), 
                       cv2.FONT_HERSHEY_DUPLEX, score_font * 0.9, (150, 150, 150), 2)
            
            combo_text = f"x{self.max_combo}"
            combo_size = cv2.getTextSize(combo_text, cv2.FONT_HERSHEY_DUPLEX, score_font * 2.5, 5)[0]
            combo_x = col2_x + (200 - combo_size[0]) // 2
            cv2.putText(frame, combo_text, (combo_x, stats_y + 120), 
                       cv2.FONT_HERSHEY_DUPLEX, score_font * 2.5, (255, 200, 0), 5)
            
            # === 2 BUTTONS: ULANGI & BERHENTI - CENTERED ===
            button_y = card_y + card_h - 140
            button_w = 300
            button_h = 90
            button_spacing = 60
            
            # Calculate centered position for buttons
            total_width = button_w * 2 + button_spacing
            btn_start_x = (self.window_width - total_width) // 2
            
            # Button ULANGI (Restart) - Hijau
            btn_restart_x = btn_start_x
            btn_restart_y = button_y
            self.button_restart = (btn_restart_x, btn_restart_y, button_w, button_h)
            
            # Check if mouse hover
            hover_restart = (btn_restart_x <= self.mouse_x <= btn_restart_x + button_w and 
                           btn_restart_y <= self.mouse_y <= btn_restart_y + button_h)
            
            restart_color = (0, 220, 0) if hover_restart else (0, 150, 0)
            cv2.rectangle(frame, (btn_restart_x, btn_restart_y), 
                         (btn_restart_x + button_w, btn_restart_y + button_h), 
                         restart_color, -1)
            cv2.rectangle(frame, (btn_restart_x, btn_restart_y), 
                         (btn_restart_x + button_w, btn_restart_y + button_h), 
                         (0, 255, 0), 5)
            
            restart_text = "ULANGI"
            text_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_DUPLEX, score_font * 1.3, 4)[0]
            text_x = btn_restart_x + (button_w - text_size[0]) // 2
            text_y = btn_restart_y + (button_h + text_size[1]) // 2
            cv2.putText(frame, restart_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, score_font * 1.3, (255, 255, 255), 4, cv2.LINE_AA)
            
            # Button BERHENTI (Quit) - Merah
            btn_quit_x = btn_start_x + button_w + button_spacing
            btn_quit_y = button_y
            self.button_quit = (btn_quit_x, btn_quit_y, button_w, button_h)
            
            # Check if mouse hover
            hover_quit = (btn_quit_x <= self.mouse_x <= btn_quit_x + button_w and 
                         btn_quit_y <= self.mouse_y <= btn_quit_y + button_h)
            
            quit_color = (0, 0, 220) if hover_quit else (0, 0, 150)
            cv2.rectangle(frame, (btn_quit_x, btn_quit_y), 
                         (btn_quit_x + button_w, btn_quit_y + button_h), 
                         quit_color, -1)
            cv2.rectangle(frame, (btn_quit_x, btn_quit_y), 
                         (btn_quit_x + button_w, btn_quit_y + button_h), 
                         (0, 0, 255), 5)
            
            quit_text = "BERHENTI"
            text_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_DUPLEX, score_font * 1.1, 4)[0]
            text_x = btn_quit_x + (button_w - text_size[0]) // 2
            text_y = btn_quit_y + (button_h + text_size[1]) // 2
            cv2.putText(frame, quit_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, score_font * 1.1, (255, 255, 255), 4, cv2.LINE_AA)
    
    def _draw_card(self, frame, x, y, w, h, bg_color, border_color=None):
        """Draw modern card with shadow"""
        # Shadow
        shadow_offset = 8
        cv2.rectangle(frame, (x + shadow_offset, y + shadow_offset), 
                     (x + w + shadow_offset, y + h + shadow_offset), (0, 0, 0), -1)
        
        # Card background
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
        
        # Border
        if border_color:
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 3)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 2)
    
    def draw_start_menu(self, frame):
        """Draw start menu screen dengan button MULAI"""
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Title dengan glow effect - BESAR
        title_font = self.window_height / 150.0
        title = "FRUIT SLICE"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_font, 8)[0]
        title_x = (self.window_width - title_size[0]) // 2
        title_y = self.window_height // 3
        
        # Glow effect
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font, (255, 150, 0), 15, cv2.LINE_AA)
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font, (0, 255, 255), 8, cv2.LINE_AA)
        
        # Subtitle
        subtitle_font = self.window_height / 600.0
        subtitle = "Hand Gesture Recognition Game"
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_DUPLEX, subtitle_font, 2)[0]
        subtitle_x = (self.window_width - subtitle_size[0]) // 2
        cv2.putText(frame, subtitle, (subtitle_x, title_y + 80), 
                   cv2.FONT_HERSHEY_DUPLEX, subtitle_font, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Instructions
        inst_font = self.window_height / 900.0
        instructions = [
            "Gunakan JARI TELUNJUK untuk memotong buah",
            "Jangan biarkan buah jatuh!",
            "Dapatkan combo untuk skor lebih tinggi!"
        ]
        inst_y = title_y + 180
        for i, text in enumerate(instructions):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, inst_font, 2)[0]
            text_x = (self.window_width - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, inst_y + i * 50), 
                       cv2.FONT_HERSHEY_DUPLEX, inst_font, (255, 255, 100), 2, cv2.LINE_AA)
        
        # Button MULAI - CENTERED & BIG
        button_w = 400
        button_h = 100
        button_x = (self.window_width - button_w) // 2
        button_y = self.window_height - 250
        self.button_start = (button_x, button_y, button_w, button_h)
        
        # Check if mouse hover
        hover = (button_x <= self.mouse_x <= button_x + button_w and 
                button_y <= self.mouse_y <= button_y + button_h)
        
        button_color = (0, 220, 0) if hover else (0, 160, 0)
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     button_color, -1)
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     (0, 255, 0), 6)
        
        # Button text
        btn_font = self.window_height / 400.0
        btn_text = "MULAI"
        text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_DUPLEX, btn_font, 5)[0]
        text_x = button_x + (button_w - text_size[0]) // 2
        text_y = button_y + (button_h + text_size[1]) // 2
        cv2.putText(frame, btn_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_DUPLEX, btn_font, (255, 255, 255), 5, cv2.LINE_AA)
    
    def reset_game(self):
        """Reset game to initial state"""
        self.fruits = []
        self.sliced_fruits = []
        self.particle_system.clear()
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.max_combo = 0
        self.game_over = False
        self.game_started = True  # Keep started
        self.hand_trail.clear()
        self.fruit_spawn_counter = 0
    
    def run(self):
        """Main game loop"""
        print("=" * 60)
        print("FRUIT SLICE GAME - HAND GESTURE RECOGNITION")
        print("=" * 60)
        print("\nInstructions:")
        print("- Gunakan jari telunjuk untuk memotong buah")
        print("- Jangan biarkan buah jatuh ke bawah!")
        print("- Tekan 'q' untuk keluar")
        print("- Tekan 'r' untuk restart\n")
        print("=" * 60)
        
        while self.running:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Resize frame to window size (responsive)
            frame = cv2.resize(frame, (self.window_width, self.window_height))
            
            # Show start menu if game not started
            if not self.game_started:
                self.draw_start_menu(frame)
            else:
                # Process hand tracking and collision detection
                self.process_hand_tracking(frame)
                
                # Only spawn and update if game is not over
                if not self.game_over:
                    # Spawn fruits periodically
                    self.fruit_spawn_counter += 1
                    if self.fruit_spawn_counter >= self.fruit_spawn_rate:
                        self.spawn_fruit()
                        self.fruit_spawn_counter = 0
                    
                    # Update game objects
                    self.update_fruits()
                
                # Draw everything
                self.draw_fruits(frame)
                self.draw_ui(frame)
            
            # Show frame (already in fullscreen from init)
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.running = False
            elif key == ord('r') or key == ord('R'):  # 'r' or 'R' to restart
                self.reset_game()
        
        # Cleanup
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nGame ditutup. Terima kasih!")
        print(f"Final Score: {self.score}")
        print(f"Max Combo: x{self.max_combo}")

def main():
    """Entry point"""
    game = FruitSliceGame()
    game.run()

if __name__ == "__main__":
    main()
