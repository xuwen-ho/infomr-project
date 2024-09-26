import pandas as pd
import matplotlib.pyplot as plt

# Assuming the CSV has headers as saved in your code
df = pd.read_csv('ShapeDatabase_INFOMR.csv')

average_vertices = df['Number of vertices'].mean()
average_faces = df['Number of faces'].mean()

print(f"Average number of vertices: {average_vertices}")
print(f"Average number of faces: {average_faces}")


plt.figure(figsize=(10, 6))
plt.hist(df['Number of vertices'], bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Vertex Counts')
plt.xlabel('Number of Vertices')
plt.ylabel('Frequency')
plt.axvline(average_vertices, color='red', linestyle='dashed', linewidth=1)
plt.text(average_vertices, plt.ylim()[1]*0.9, 'Average', color = 'red')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['Number of faces'], bins=20, color='green', edgecolor='black')
plt.title('Histogram of Face Counts')
plt.xlabel('Number of Faces')
plt.ylabel('Frequency')
plt.axvline(average_faces, color='red', linestyle='dashed', linewidth=1)
plt.text(average_faces, plt.ylim()[1]*0.9, 'Average', color = 'red')
plt.show()


# Identifying outliers for vertices
Q1_vertices = df['Number of vertices'].quantile(0.25)
Q3_vertices = df['Number of vertices'].quantile(0.75)
IQR_vertices = Q3_vertices - Q1_vertices
outliers_vertices = df[(df['Number of vertices'] < (Q1_vertices - 1.5 * IQR_vertices)) | (df['Number of vertices'] > (Q3_vertices + 1.5 * IQR_vertices))]

# Identifying outliers for faces
Q1_faces = df['Number of faces'].quantile(0.25)
Q3_faces = df['Number of faces'].quantile(0.75)
IQR_faces = Q3_faces - Q1_faces
outliers_faces = df[(df['Number of faces'] < (Q1_faces - 1.5 * IQR_faces)) | (df['Number of faces'] > (Q3_faces + 1.5 * IQR_faces))]

print("Vertex Outliers:")
print(outliers_vertices)
print("Face Outliers:")
print(outliers_faces)

# Concatenate outliers into a single DataFrame and remove duplicates
all_outliers = pd.concat([outliers_vertices, outliers_faces]).drop_duplicates()

print("All Outliers Combined:")
print(all_outliers)


