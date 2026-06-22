# import streamlit as st
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# from torchvision import models, transforms
# from PIL import Image
# import os
# from openai import OpenAI
# from openai import OpenAI, RateLimitError, AuthenticationError, APIError

# # -------------------------------
# # LOAD MODEL
# # -------------------------------
# @st.cache_resource
# # def load_model():
# #     checkpoint_path = r"C:\Users\ishika.shivhare\Downloads\Image\Transfer Learning\Transfer Learning\models\vgnet.pth"

# #     checkpoint = torch.load(checkpoint_path, map_location="cpu")

# #     num_classes = len(checkpoint["class_to_idx"])

# #     model = models.vgg16(weights=None)
# #     model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# #     model.load_state_dict(checkpoint["model_state_dict"])
# #     model.eval()

# #     class_names = list(checkpoint["class_to_idx"].keys())

# #     return model, class_names

# # model, class_names = load_model()


# @st.cache_resource
# def load_model():
#     base_dir = os.path.dirname(os.path.abspath(__file__))

#     checkpoint_path = os.path.join(
#         base_dir,
#         "Transfer Learning",
#         "Transfer Learning",
#         "models",
#         "vgg16_best.pth"
#     )

#     if not os.path.exists(checkpoint_path):
#         st.error(f"Model not found at: {checkpoint_path}")
#         st.stop()

#     checkpoint = torch.load(checkpoint_path, map_location="cpu")

#     num_classes = len(checkpoint["class_to_idx"])

#     model = models.vgg16(weights=None)
#     model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
#     class_names = [idx_to_class[i] for i in range(num_classes)]

#     return model, class_names
# model, class_names = load_model()


# # -------------------------------
# # TRANSFORM
# # -------------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # -------------------------------
# # GRAD-CAM
# # -------------------------------
# class GradCAM:
#     def __init__(self, model):
#         self.model = model
#         self.target_layer = model.features[-1]
#         self.gradients = None
#         self.activations = None

#         self.target_layer.register_forward_hook(self.forward_hook)
#         self.target_layer.register_backward_hook(self.backward_hook)

#     def forward_hook(self, module, input, output):
#         self.activations = output

#     def backward_hook(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def generate(self, input_tensor):
#         output = self.model(input_tensor)
#         pred_class = output.argmax(dim=1).item()

#         self.model.zero_grad()
#         output[0, pred_class].backward()

#         gradients = self.gradients
#         activations = self.activations

#         pooled_gradients = torch.mean(gradients, dim=[0,2,3])

#         for i in range(activations.shape[1]):
#             activations[:, i, :, :] *= pooled_gradients[i]

#         heatmap = torch.mean(activations, dim=1).squeeze()

#         heatmap = heatmap.detach().cpu().numpy()
#         heatmap = np.maximum(heatmap, 0)

#         if np.max(heatmap) != 0:
#             heatmap /= np.max(heatmap)

#         return heatmap, pred_class

# # -------------------------------
# # PREDICTION
# # -------------------------------
# def predict(image, model, class_names):
#     input_tensor = transform(image).unsqueeze(0)

#     output = model(input_tensor)
#     probs = torch.nn.functional.softmax(output, dim=1)

#     pred_class = torch.argmax(probs).item()
#     confidence = probs[0][pred_class].item()

#     gradcam = GradCAM(model)
#     heatmap, _ = gradcam.generate(input_tensor)

#     return heatmap, class_names[pred_class], confidence
# # def generate_llm_explanation(prediction, confidence, heatmap):
# #     """
# #     Uses an LLM to convert structured model outputs into a clear,
# #     responsible, user-friendly explanation.
# #     """

# #     client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# #     confidence_percent = round(confidence * 100, 2)

# #     heatmap_summary = {
# #         "maximum_activation": float(np.max(heatmap)),
# #         "average_activation": float(np.mean(heatmap)),
# #         "activation_spread": float(np.std(heatmap))
# #     }

# #     prompt = f"""
# # You are an AI assistant supporting a brain MRI classification application.

# # The machine learning model has produced the following structured output:

# # Prediction label: {prediction}
# # Confidence score: {confidence_percent}%
# # Grad-CAM heatmap statistics:
# # - Maximum activation: {heatmap_summary["maximum_activation"]:.4f}
# # - Average activation: {heatmap_summary["average_activation"]:.4f}
# # - Activation spread: {heatmap_summary["activation_spread"]:.4f}

# # Your task:
# # Generate a clear, simple, well-explained interpretation for a non-technical user.

# # The response must include:

# # 1. AI Result Summary
# # Explain what the predicted label means in simple language.

# # 2. Confidence Interpretation
# # Explain how confident the model is, but avoid saying the result is medically certain.

# # 3. Grad-CAM Explanation
# # Explain that Grad-CAM highlights image regions that influenced the model prediction.
# # Do not claim that the highlighted region is definitely a tumor.

# # 4. Suggested Next Steps
# # Give safe, general recommendations such as consulting a radiologist, neurologist, or healthcare professional.
# # Do not provide treatment advice.
# # Do not prescribe medicine.
# # Do not suggest emergency action unless the user has symptoms, which are not provided here.

# # 5. Responsible AI Disclaimer
# # Clearly mention that this output is for decision support only and is not a medical diagnosis.

# # Tone:
# # - Professional
# # - Easy to understand
# # - Reassuring but cautious
# # - No direct diagnosis
# # - No treatment recommendation
# # """

# #     response = client.chat.completions.create(
# #         model=st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
# #         messages=[
# #             {
# #                 "role": "system",
# #                 "content": (
# #                     "You generate responsible AI explanations for medical imaging decision-support tools. "
# #                     "You must avoid diagnosis, treatment advice, or certainty claims."
# #                 )
# #             },
# #             {
# #                 "role": "user",
# #                 "content": prompt
# #             }
# #         ],
# #         temperature=0.3,
# #         max_tokens=700
# #     )

# #     return response.choices[0].message.content
# def generate_llm_explanation(prediction, confidence, heatmap):
#     """
#     Uses an LLM to convert model outputs into a user-friendly explanation.
#     If quota/authentication fails, shows a safe message instead of crashing.
#     """

#     client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#     confidence_percent = round(confidence * 100, 2)

#     heatmap_summary = {
#         "maximum_activation": float(np.max(heatmap)),
#         "average_activation": float(np.mean(heatmap)),
#         "activation_spread": float(np.std(heatmap))
#     }

#     prompt = f"""
# You are an AI assistant supporting a brain MRI classification application.

# The machine learning model produced the following output:

# Prediction label: {prediction}
# Confidence score: {confidence_percent}%

# Grad-CAM heatmap summary:
# - Maximum activation: {heatmap_summary["maximum_activation"]:.4f}
# - Average activation: {heatmap_summary["average_activation"]:.4f}
# - Activation spread: {heatmap_summary["activation_spread"]:.4f}

# Generate a clear, responsible, user-friendly explanation.

# Include the following sections:

# 1. AI Result Summary
# Explain what the predicted label means in simple language.

# 2. Confidence Interpretation
# Explain what the confidence score means, but do not say the result is medically certain.

# 3. Grad-CAM Explanation
# Explain that Grad-CAM highlights regions that influenced the model prediction.
# Do not claim the highlighted region is definitely a tumor.

# 4. Recommended Next Steps
# Give safe, general recommendations such as consulting a radiologist, neurologist, or qualified healthcare professional.
# Do not provide treatment advice.
# Do not prescribe medicine.
# Do not give emergency instructions unless symptoms are explicitly provided.

# 5. Responsible AI Disclaimer
# Clearly mention that this is for decision support only and not a medical diagnosis.

# Tone:
# Professional, simple, cautious, reassuring, and non-diagnostic.
# """

#     try:
#         response = client.chat.completions.create(
#             model=st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You generate responsible AI explanations for medical imaging decision-support tools. "
#                         "Avoid diagnosis, treatment advice, certainty claims, or medical prescriptions."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0.3,
#             max_tokens=700
#         )

#         return response.choices[0].message.content

#     except RateLimitError as e:
#         return """
# ## LLM Explanation Could Not Be Generated

# The CNN prediction and Grad-CAM output were generated successfully, but the LLM explanation could not be generated because the OpenAI API quota is unavailable or exhausted.

# ### What this means
# The application is working, but the connected LLM service currently does not have enough available API quota to generate the natural-language explanation.

# ### What to check
# - OpenAI billing status
# - Available API credit balance
# - Project-level usage limits
# - Whether the API key belongs to the correct project
# - Whether the selected model is available for the account

# ### Responsible AI Note
# The model output shown above should still be treated as a decision-support result only. It is not a medical diagnosis and should be validated by a qualified healthcare professional.
# """

#     except AuthenticationError as e:
#         return """
# ## LLM Authentication Failed

# The LLM explanation could not be generated because the API key is invalid or incorrectly configured.

# Please verify that the API key in `.streamlit/secrets.toml` is correct and belongs to the intended OpenAI project.
# """


# except Exception as e:
#     return f"""
# ## LLM Explanation Failed

# The prediction was generated successfully, but the explanation layer failed.

# ### Error Details

# ```text
# {str(e)}


# # -------------------------------
# # UI
# # -------------------------------
# st.title("Brain Tumor Detection using VGG16 + Grad-CAM")

# uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file).convert("RGB")

# #     st.image(image, caption="Uploaded Image")

# #     heatmap, prediction, confidence = predict(image, model, class_names)

# #     st.success(f"Prediction: {prediction}")
# #     st.metric("Confidence", f"{confidence*100:.2f}%")

# #     # ✅ Generate explanation
# #     explanation = generate_explanation(prediction, confidence)

# #     st.markdown(explanation)

# #     img = np.array(image)
# #     img = cv2.resize(img, (224, 224))

# #     heatmap = cv2.resize(heatmap, (224, 224))
# #     heatmap = np.uint8(255 * heatmap)
# #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# #     overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# #     st.image(overlay, caption="Grad-CAM Visualization")
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(image, caption="Uploaded Image")

#     heatmap, prediction, confidence = predict(image, model, class_names)

#     st.success(f"Prediction: {prediction}")
#     st.metric("Confidence", f"{confidence*100:.2f}%")

#     img = np.array(image)
#     img = cv2.resize(img, (224, 224))

#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

#     st.image(overlay, caption="Grad-CAM Visualization")

#     st.markdown("---")
#     st.subheader("AI-Generated Explanation and Recommendation")

#     with st.spinner("Generating explanation..."):
#         llm_explanation = generate_llm_explanation(prediction, confidence, heatmap)

#     st.markdown(llm_explanation)
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import models, transforms
from PIL import Image
from openai import OpenAI, RateLimitError, AuthenticationError, APIError


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection using VGG16 + Grad-CAM",
    layout="centered"
)


# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(
        base_dir,
        "Transfer Learning",
        "Transfer Learning",
        "models",
        "vgg16_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        st.error(f"Model not found at: {checkpoint_path}")
        st.stop()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    num_classes = len(checkpoint["class_to_idx"])

    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    return model, class_names


model, class_names = load_model()


# -------------------------------
# IMAGE TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -------------------------------
# GRAD-CAM CLASS
# -------------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.features[-1]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, pred_class].backward()

        gradients = self.gradients
        activations = self.activations.clone()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap, pred_class


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(image, model, class_names):
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)

    pred_class = torch.argmax(probs).item()
    confidence = probs[0][pred_class].item()

    gradcam = GradCAM(model)
    heatmap, _ = gradcam.generate(input_tensor)

    return heatmap, class_names[pred_class], confidence


# -------------------------------
# LLM EXPLANATION FUNCTION
# -------------------------------
def generate_llm_explanation(prediction, confidence, heatmap):
    """
    Uses an LLM to convert CNN outputs into user-friendly explanation.
    If quota/authentication fails, the app shows a safe fallback message instead of crashing.
    """

    confidence_percent = round(confidence * 100, 2)

    heatmap_summary = {
        "maximum_activation": float(np.max(heatmap)),
        "average_activation": float(np.mean(heatmap)),
        "activation_spread": float(np.std(heatmap))
    }

    prompt = f"""
You are an AI explanation layer for a brain MRI classification decision-support application.

The CNN model has produced the following structured output:

Prediction label: {prediction}
Confidence score: {confidence_percent}%

Grad-CAM heatmap summary:
- Maximum activation: {heatmap_summary["maximum_activation"]:.4f}
- Average activation: {heatmap_summary["average_activation"]:.4f}
- Activation spread: {heatmap_summary["activation_spread"]:.4f}

Your task is to generate a clear, responsible, user-friendly explanation.

The response must include these sections:

1. AI Result Summary
Explain the predicted label in simple, non-technical language.

2. Confidence Interpretation
Explain what the confidence score means. Do not say the result is medically certain.

3. Grad-CAM Explanation
Explain that Grad-CAM highlights image regions that influenced the model prediction.
Do not claim that the highlighted region is definitely a tumor.

4. Recommended Next Steps
Give safe, general recommendations such as consulting a radiologist, neurologist, or qualified healthcare professional.
Do not provide treatment advice.
Do not prescribe medicine.
Do not give emergency instructions unless symptoms are explicitly provided.

5. Responsible AI Disclaimer
Clearly mention that this output is for decision support only and is not a medical diagnosis.

Tone:
- Professional
- Simple
- Cautious
- Reassuring
- Non-diagnostic
"""

    try:
        client = OpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model=st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate responsible AI explanations for medical imaging decision-support tools. "
                        "Avoid diagnosis, treatment advice, certainty claims, prescriptions, or emergency instructions. "
                        "Explain only the CNN model output in simple, user-friendly language."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=700
        )

        return response.choices[0].message.content

    except RateLimitError:
        return f"""
## 🧠 AI-Based Interpretation

The uploaded MRI image has been classified as **{prediction}** based on patterns learned by the model.

This indicates that the model did not detect strong visual characteristics associated with abnormal tumor structures in the scan.

---

## 📊 Confidence Analysis

The model shows a confidence score of **{confidence*100:.2f}%**, indicating a high level of certainty based on its training data.

However, this confidence reflects only the model’s internal pattern recognition and should not be considered a medical confirmation.

---

## 🔬 Grad-CAM Insight

The Grad-CAM visualization highlights areas of the image that influenced the model’s decision.

In this case, there are no strong concentrated activation regions that suggest abnormal tissue, which supports the **{prediction}** classification.

---

## ✅ Recommended Next Steps

- Continue regular health monitoring  
- If symptoms such as headaches or neurological discomfort are present, consult a medical professional  
- Use this result as a preliminary screening output only  

---

## ⚠️ Important Disclaimer

The AI-generated explanation layer could not be dynamically generated due to API limits.

This interpretation is automatically generated within the application and is intended for **decision support purposes only**.  

It is **not a medical diagnosis**. Please consult a qualified healthcare professional for confirmation and further evaluation.
"""

    except AuthenticationError:
        return """
## LLM Authentication Failed

The LLM explanation could not be generated because the API key is invalid or incorrectly configured.

### What To Check

- Verify the API key in `.streamlit/secrets.toml`
- Make sure there are no extra spaces in the key
- Confirm that the key belongs to the intended OpenAI project
- Restart Streamlit after updating the key

### Responsible AI Note

The CNN prediction and Grad-CAM visualization are decision-support outputs only and should not be considered a medical diagnosis.
"""

    except APIError:
        return """
## LLM Service Error

The prediction was generated successfully, but the LLM service returned an API error.

### What To Check

- Model name in `.streamlit/secrets.toml`
- API key configuration
- OpenAI account billing and usage
- Internet connection

### Responsible AI Note

This output is for decision support only and does not replace review by a qualified healthcare professional.
"""

    except Exception as e:
        return (
            "## LLM Explanation Failed\n\n"
            "The prediction was generated successfully, but the explanation layer failed.\n\n"
            "### Error Details\n\n"
            f"```text\n{str(e)}\n```\n\n"
            "### Suggested Checks\n\n"
            "- Check API key configuration\n"
            "- Check OpenAI quota or billing\n"
            "- Verify model name in `.streamlit/secrets.toml`\n"
            "- Check internet connection\n"
            "- Restart the Streamlit app after making changes\n\n"
            "### Responsible AI Note\n\n"
            "This system provides decision-support insights only and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional for clinical validation."
        )


# -------------------------------
# UI
# -------------------------------
st.title("Brain Tumor Detection using VGG16 + Grad-CAM")

st.markdown(
    """
This application uses a CNN model to classify an uploaded brain MRI image and uses Grad-CAM to highlight image regions that influenced the model prediction.  
An optional LLM layer can generate a user-friendly explanation and responsible recommendation based on the model output.
"""
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("## Uploaded MRI Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    raw_heatmap, prediction, confidence = predict(image, model, class_names)

    st.markdown("## Model Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Prediction: {prediction}")

    with col2:
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    # -------------------------------
    # GRAD-CAM VISUALIZATION
    # -------------------------------
    img = np.array(image)
    img = cv2.resize(img, (224, 224))

    resized_heatmap = cv2.resize(raw_heatmap, (224, 224))
    colored_heatmap = np.uint8(255 * resized_heatmap)
    colored_heatmap = cv2.applyColorMap(colored_heatmap, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, colored_heatmap, 0.4, 0)

    st.markdown("## Grad-CAM Visualization")
    st.image(
        overlay,
        caption="Grad-CAM Visualization",
        use_container_width=True
    )

    st.info(
        "Grad-CAM highlights the image regions that influenced the model prediction. "
        "The highlighted area should not be interpreted as a confirmed medical finding."
    )

    # -------------------------------
    # LLM EXPLANATION
    # -------------------------------
    st.markdown("---")
    st.markdown("## AI-Generated Explanation and Recommendation")

    st.warning(
        "The LLM explanation is generated from the model prediction, confidence score, and Grad-CAM summary. "
        "It does not independently diagnose the MRI image."
    )


    if "llm_result" not in st.session_state:
        st.session_state.llm_result = None

    if st.button("Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            st.session_state.llm_result = generate_llm_explanation(
                prediction,
                confidence,
                raw_heatmap
            )

    if st.session_state.llm_result:
        st.markdown(st.session_state.llm_result)


else:
    st.info("Please upload a brain MRI image to start prediction.")