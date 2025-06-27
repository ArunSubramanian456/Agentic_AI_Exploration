
import base64


# Helper Function: Encode Image to Base64
def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base64 string.
    Required for sending local images to OpenAI's vision models.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        print(f"Image '{image_path}' encoded to base64.")
        return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None