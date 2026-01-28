from PIL import Image

def resize_and_save(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    
    # Berechne Skalierung um Aspect Ratio beizubehalten
    scale = min(512 / w, 288 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize mit beibehaltener Ratio
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Erstelle neues Bild mit schwarzem Hintergrund (512x288)
    new_img = Image.new("RGB", (512, 288), (0, 0, 0))
    
    # Zentriere das resized Bild
    left = (512 - new_w) // 2
    top = (288 - new_h) // 2
    
    new_img.paste(resized, (left, top))
    new_img.save(output_path)

if __name__ == "__main__":
    resize_and_save("bienen.jpg", "bienen_resized.jpg")


