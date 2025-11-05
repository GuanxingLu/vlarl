import numpy as np
import cv2
from PIL import Image


def add_info_board(img, **kwargs):
    """
    Add a white information board to the right of an image with flexible content.
    
    Args:
        img: Input image (numpy array or PIL Image)
        **kwargs: Variable key-value pairs to display on the board
        
    Returns:
        combined_img: Image with info board added
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    # Fixed parameters
    board_width = 280
    board_height = img.shape[0]
    line_height = 15
    character_limit = 35
    margin_left = img.shape[1] + 10
    
    # Create white board and combine with image
    board = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255
    combined_img = np.hstack((img, board))
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_color = (0, 0, 0)
    
    # Draw title
    # cv2.putText(combined_img, "States", (margin_left, 30),
    #             font, font_scale + 0.0, text_color, 2, cv2.LINE_AA)
    # y_position = 50     # the y position of the first line
    y_position = 10
    
    # Process each key-value pair
    for key, value in kwargs.items():
        if value is None:
            continue
        # Draw key
        cv2.putText(combined_img, f"{key}:", (margin_left, y_position),
                    font, font_scale, text_color, 1, cv2.LINE_AA)
        y_position += line_height
        
        # Convert value to string and split into lines
        value_str = str(value)
        lines = []
        words = value_str.split()
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > character_limit:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw value lines
        for line in lines:
            cv2.putText(combined_img, line, (margin_left, y_position),
                       font, font_scale, text_color, 1, cv2.LINE_AA)
            y_position += line_height
            
        # Add blank line between different inputs
        # y_position += line_height
        
        # Check if we're running out of vertical space
        if y_position >= board_height - line_height:
            break
            
    return combined_img