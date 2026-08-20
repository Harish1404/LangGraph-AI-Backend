import io
import requests
from PIL import Image
from litellm import image_generation
from app.core.config import settings

async def generate_image(prompt: str) -> Image.Image:
    """
    Generates an image based on the prompt using Gemini's image generation model,
    downloads it, and returns a PIL Image object.
    """
    # 1. Call LiteLLM image generation API
    # Note: We use 'gemini/imagen-3.0-generate-002' as the model name
    response = image_generation(
        model="gemini/Gemini 2.5 Flash Preview Image",
        prompt=prompt,
        api_key=settings.gemini_api_key,
        n=1
    )
    
    # 2. Extract the URL from the response (returns ImageResponse object)
    image_url = response.data[0].url
    
    # 3. Download the image bytes
    img_response = requests.get(image_url)
    img_response.raise_for_status()
    
    # 4. Open the image with Pillow and return it
    image = Image.open(io.BytesIO(img_response.content))
    return image

if __name__ == "__main__":
    import os
    import asyncio
    
    async def main():
        prompt = "A futuristic flying car over a sci-fi city"
        print(f"Generating image for prompt: '{prompt}'...")
        try:
            image = await generate_image(prompt)
            
            # Ensure the outputs directory exists
            os.makedirs("outputs", exist_ok=True)
            output_path = "outputs/generated_image.png"
            
            # Save the PIL Image
            image.save(output_path)
            print(f"\nSuccess! You can find the generated image at:\n{os.path.abspath(output_path)}")
        except Exception as e:
            print(f"Error during image generation: {e}")
            
    asyncio.run(main())