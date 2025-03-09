import streamlit as st # type: ignore
import torch # type: ignore
from torchvision import models, transforms # type: ignore
from PIL import Image # type: ignore
import torch.nn as nn # type: ignore
#import matplotlib.pyplot as plt # type: ignore


#warnings.simplefilter("ignore", category=ComplexWarning)
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Medical Image Analysis to detect Brain Tumor </h2>", unsafe_allow_html=True)



class_labels = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor"
}


model = models.resnet50(pretrained=False)


model.fc = nn.Linear(model.fc.in_features, 4)



model.load_state_dict(torch.load(r'C:\Users\POORNIMA\Desktop\POORNIMA\DataScienceProjects\Final Project\resnet50_brain_tumor.pth', weights_only=True), strict=False)


model.eval()

t_pipeline = transforms.Compose([ 
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
   transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
  transforms.Resize((224, 224)),          
      transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],) 
])


uploaded_image = st.file_uploader("Please Upload an image", type=["jpg", "png", "jpeg"])


   

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)
     

    #st.image(image, caption="Uploaded Image")
    image_tensor = t_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class_index = torch.max(output, 1)
        predicted_label = class_labels[predicted_class_index.item()]



        #st.write(f'Predicted Tumor: {predicted_label}')
       
        st.markdown(f"""
    <div style="text-align: center;">
        <strong><span style='color: red;'>Predicted Tumor: {predicted_label}</span></strong>
    </div>
""", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
       
else:
      st.warning("Please upload an image!")



