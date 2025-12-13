import cv2
import random
import numpy as np

class Particle:
    """Particle effect for sliced fruits"""
    
    def __init__(self, x, y, color):
        """Initialize particle at position (x, y) with color"""
        self.x = x
        self.y = y
        self.color = color
        
        # Random velocity
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 8)
        self.velocity_x = np.cos(angle) * speed
        self.velocity_y = np.sin(angle) * speed - random.uniform(2, 5)  # Upward bias
        
        # Properties
        self.size = random.randint(3, 8)
        self.life = 1.0  # 0 to 1
        self.decay_rate = random.uniform(0.02, 0.05)
        self.gravity = 0.3
    
    def update(self):
        """Update particle position and life"""
        # Apply velocity
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Apply gravity
        self.velocity_y += self.gravity
        
        # Decay life
        self.life -= self.decay_rate
        
        # Slow down
        self.velocity_x *= 0.98
        self.velocity_y *= 0.98
    
    def draw(self, frame):
        """Draw particle on frame"""
        if self.life <= 0:
            return
        
        # Calculate alpha based on life
        alpha = int(self.life * 255)
        
        # Calculate size based on life
        current_size = int(self.size * self.life)
        
        if current_size < 1:
            return
        
        # Draw circle
        x, y = int(self.x), int(self.y)
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x, y), current_size, self.color, -1)
    
    def is_dead(self):
        """Check if particle should be removed"""
        return self.life <= 0


class ParticleSystem:
    """Manage multiple particles"""
    
    def __init__(self):
        self.particles = []
    
    def emit(self, x, y, color, count=20):
        """Emit particles at position"""
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
    
    def update(self):
        """Update all particles"""
        # Update each particle
        for particle in self.particles:
            particle.update()
        
        # Remove dead particles
        self.particles = [p for p in self.particles if not p.is_dead()]
    
    def draw(self, frame):
        """Draw all particles"""
        for particle in self.particles:
            particle.draw(frame)
    
    def clear(self):
        """Clear all particles"""
        self.particles.clear()


class SlicedFruit:
    """Sliced fruit halves with animation"""
    
    def __init__(self, fruit, slice_angle=0):
        """Initialize sliced fruit from original fruit"""
        self.x = fruit.x
        self.y = fruit.y
        self.radius = fruit.radius
        self.image = fruit.image
        self.name = fruit.name
        
        # Split into two halves
        self.slice_angle = slice_angle
        
        # Left half
        self.left_x = fruit.x - 10
        self.left_y = fruit.y
        self.left_velocity_x = -random.uniform(3, 6)
        self.left_velocity_y = -random.uniform(2, 5)
        self.left_rotation = random.uniform(-10, -5)
        self.left_angle = 0
        
        # Right half
        self.right_x = fruit.x + 10
        self.right_y = fruit.y
        self.right_velocity_x = random.uniform(3, 6)
        self.right_velocity_y = -random.uniform(2, 5)
        self.right_rotation = random.uniform(5, 10)
        self.right_angle = 0
        
        # Life
        self.life = 1.0
        self.decay_rate = 0.02
        self.gravity = 0.4
    
    def update(self):
        """Update sliced fruit halves"""
        # Update left half
        self.left_x += self.left_velocity_x
        self.left_y += self.left_velocity_y
        self.left_velocity_y += self.gravity
        self.left_angle += self.left_rotation
        
        # Update right half
        self.right_x += self.right_velocity_x
        self.right_y += self.right_velocity_y
        self.right_velocity_y += self.gravity
        self.right_angle += self.right_rotation
        
        # Decay life
        self.life -= self.decay_rate
    
    def draw(self, frame):
        """Draw sliced fruit halves"""
        if self.life <= 0 or self.image is None:
            return
        
        alpha = self.life
        
        # Draw left half
        self._draw_half(frame, self.left_x, self.left_y, self.left_angle, alpha, is_left=True)
        
        # Draw right half
        self._draw_half(frame, self.right_x, self.right_y, self.right_angle, alpha, is_left=False)
    
    def _draw_half(self, frame, x, y, angle, alpha, is_left):
        """Draw one half of sliced fruit"""
        if self.image is None:
            return
        
        # Get original image dimensions
        orig_h, orig_w = self.image.shape[:2]
        
        # Calculate new dimensions
        size = self.radius * 2
        aspect_ratio = orig_w / orig_h
        
        if aspect_ratio > 1:
            new_w = size
            new_h = int(size / aspect_ratio)
        else:
            new_h = size
            new_w = int(size * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Crop to half
        if is_left:
            resized = resized[:, :new_w//2]
        else:
            resized = resized[:, new_w//2:]
        
        # Calculate position
        half_w = resized.shape[1]
        half_h = resized.shape[0]
        
        x1 = int(x - half_w // 2)
        y1 = int(y - half_h // 2)
        x2 = x1 + half_w
        y2 = y1 + half_h
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return
        
        # Overlay with alpha
        if resized.shape[2] == 4:
            bgr = resized[:, :, :3]
            img_alpha = (resized[:, :, 3:4] / 255.0) * alpha
            
            roi = frame[y1:y2, x1:x2]
            blended = (img_alpha * bgr + (1 - img_alpha) * roi).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
    
    def is_dead(self):
        """Check if sliced fruit should be removed"""
        return self.life <= 0 or self.left_y > 2000 or self.right_y > 2000
