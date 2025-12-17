import random
import cv2
import numpy as np
import os

class Fruit:
    # Fruit types with image paths
    FRUIT_TYPES = {
        'straw': {'image': 'images/straw.png', 'name': '🍎'},
        'orange': {'image': 'images/jeruk.png', 'name': '🍊'},
        'banana': {'image': 'images/pisang.png', 'name': '🍌'},
        'watermelon': {'image': 'images/semangka.png', 'name': '🍉'},
        'apel': {'image': 'images/apel.png', 'name': '🍎'},
        'nanas': {'image': 'images/nanas.png', 'name': '�'},
    }
    
    # Cache loaded images
    _image_cache = {}
    
    def __init__(self, screen_width, screen_height):
        """Initialize fruit with random properties"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Choose random fruit type
        self.fruit_type = random.choice(list(self.FRUIT_TYPES.keys()))
        self.image_path = self.FRUIT_TYPES[self.fruit_type]['image']
        self.name = self.FRUIT_TYPES[self.fruit_type]['name']
        
        # Load image
        self.image = self._load_image(self.image_path)
        
        # Size - Apel, nanas, pisang lebih besar
        if self.fruit_type in ['apel', 'nanas', 'banana']:
            self.radius = random.randint(120, 140)  # Lebih besar
        else:
            self.radius = random.randint(80, 120)  # Ukuran normal
        
        # Starting position (random X, start from top)
        self.x = random.randint(self.radius, screen_width - self.radius)
        self.y = -self.radius  # Start above screen
        
        # Velocity - Lebih lambat dan smooth
        self.velocity_y = random.uniform(3, 6)   # Falling speed (lebih lambat)
        self.velocity_x = random.uniform(-2, 2)  # Horizontal drift (lebih smooth)
        
        # Rotation (for visual effect)
        self.rotation = 0
        self.rotation_speed = random.uniform(-5, 5)
        
        # State
        self.is_sliced = False
    
    @classmethod
    def _load_image(cls, image_path):
        """Load and cache fruit image with transparency"""
        # Check if image is already cached
        if image_path in cls._image_cache:
            return cls._image_cache[image_path]
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
        
        # Load image with alpha channel (transparency)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            return None
        
        # Cache the image
        cls._image_cache[image_path] = image
        return image
    
    def update(self):
        """Update fruit position"""
        # Update position
        self.y += self.velocity_y
        self.x += self.velocity_x
        
        # Update rotation
        self.rotation += self.rotation_speed
        
        # Add slight gravity effect
        self.velocity_y += 0.1
        
        # Bounce off side walls
        if self.x <= self.radius or self.x >= self.screen_width - self.radius:
            self.velocity_x *= -0.8
            self.x = max(self.radius, min(self.x, self.screen_width - self.radius))
    
    def draw(self, frame):
        """Draw fruit on frame"""
        if self.image is None:
            # Fallback: draw simple circle if image not available
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, 
                      (0, 255, 0), -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, 
                      (255, 255, 255), 2)
            return
        
        # Get original image dimensions
        orig_h, orig_w = self.image.shape[:2]
        
        # Calculate new dimensions maintaining aspect ratio
        size = self.radius * 2
        aspect_ratio = orig_w / orig_h
        
        if aspect_ratio > 1:  # Width > Height
            new_w = size
            new_h = int(size / aspect_ratio)
        else:  # Height >= Width
            new_h = size
            new_w = int(size * aspect_ratio)
        
        # Resize image with high-quality interpolation
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate position (centered)
        x1 = int(self.x - new_w // 2)
        y1 = int(self.y - new_h // 2)
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # Clip to frame boundaries
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return  # Skip if out of bounds
        
        # Overlay image with transparency
        if resized.shape[2] == 4:  # Has alpha channel
            # Extract RGB and alpha channels
            bgr = resized[:, :, :3]
            alpha = resized[:, :, 3:4] / 255.0
            
            # Get the region of interest from frame
            roi = frame[y1:y2, x1:x2]
            
            # Blend image with frame using alpha channel (vectorized)
            blended = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
        else:
            # No transparency, just paste
            frame[y1:y2, x1:x2] = resized
    
    def is_out_of_screen(self):
        """Check if fruit is out of screen bounds"""
        return self.y - self.radius > self.screen_height
    
    def get_position(self):
        """Return current position"""
        return (int(self.x), int(self.y))
    
    def get_bounds(self):
        """Return bounding box for collision detection"""
        return {
            'x': self.x,
            'y': self.y,
            'radius': self.radius
        }
    
    def check_line_collision(self, p1, p2):
        """Check if line segment (p1, p2) intersects with fruit circle"""
        if self.is_sliced:
            return False
        
        # Line segment from p1 to p2
        x1, y1 = p1
        x2, y2 = p2
        
        # Circle center and radius
        cx, cy = self.x, self.y
        r = self.radius
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from p1 to circle center
        fx = x1 - cx
        fy = y1 - cy
        
        # Quadratic equation coefficients
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - r * r
        
        # Check if line is too short
        if a < 0.001:
            return False
        
        # Calculate discriminant
        discriminant = b * b - 4 * a * c
        
        # No intersection
        if discriminant < 0:
            return False
        
        # Calculate intersection points
        discriminant = discriminant ** 0.5
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # Check if intersection is within line segment (0 <= t <= 1)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1):
            return True
        
        return False
    
    def slice(self):
        """Mark fruit as sliced"""
        self.is_sliced = True
