from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dotenv import load_dotenv
#HF_TOKEN=hf_*******************************
load_dotenv() 

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "If the ship sets sail it will sink!",
    "My mom loves ME!!",
    "The wet dog sighs by the dry fire.",
    "I don't know enough math for this class, but SQL I can do.",
    "The quick brown fox jumped over the lazy dog",
    "The Chicago Bears will win the Superbowl, this season",
    "Professor Posey is among the best professors that teach at BYU at 3:30 pm",
    "This phishing email almost got me",
    "The family's trip to Yellowstone was NOTHING like the TV series",
    "I'm thinking of a kite in the wind while my teacher is at the front of the class.",
    "That after they should be destroyed, even that great city Jerusalem, and many be carried away captive into Babylon, according to the own due time of the Lord, they should return again, yea, even be brought back out of captivity; and after they should be brought back out of captivity they should possess again the land of their inheritance.",
    "Therefore remember, O man, for all thy doings thou shalt be brought into judgment."
]

# Step 1: Generate embeddings for our 'documents'
embeddings = model.encode(texts)

# Confirm the shape of embeddings
print(f"Embeddings array shape: {embeddings.shape}")
print(f"Number of sentences encoded: {embeddings.shape[0]}")
print(f"Dimensions per embedding: {embeddings.shape[1]}")

# Step 2a: Inspect the embeddings
for i, emb in enumerate(embeddings):
    print(f"Text {i} preview: ", np.round(emb[:5], 3))

# Step 2b: Visualize the embeddings' similarities in 2d space 
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(6, 4))
plt.scatter(reduced[:, 0], reduced[:, 1])
for i, text in enumerate(texts):
    plt.annotate(f"Text {i}: {text}", (reduced[i, 0], reduced[i, 1]))
plt.title("Embeddings Visualized (PCA)")
plt.show()
"""
# Step 2b: Visualize the embeddings' similarities in 3D space
pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2])

for i, text in enumerate(texts):
    ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], f"Text {i}: {text}")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Embeddings Visualized (PCA, 3D)")
plt.show()
"""