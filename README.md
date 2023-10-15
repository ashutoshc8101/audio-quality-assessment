<!DOCTYPE html>
<html>

<head>
   
</head>

<body>

<h1>Audio Quality Assessment with Transformer-Based Learning</h1>

<p>This GitHub project introduces a novel approach to audio quality assessment using transformer-based deep learning architecture. The proposed model leverages the power of transformers to process audio data, providing enhanced performance over traditional approaches. This README provides an overview of the architecture, model configuration, and the tools used for this project.</p>



<h2>Architecture Overview and Model Configuration</h2>

<p>The proposed model employs a transformer-based deep learning approach to assess audio quality. It takes hand-crafted features concatenated into a vector as input and is trained with corresponding ground-truth labels. The transformer architecture, comprising an Encoder-Decoder structure with Multi-Head Attention (MHA) and Feed-Forward layers, processes the data. We utilized four layers of the encoder, set the number of heads (h) in each MHA to four, and employed an Adam optimizer. The model outputs a single continuous value representing audio quality in the range of 1 to 5. These design choices optimize feature vectors while considering attention mechanisms for enhanced performance.</p>
<p>We integrated the Dual Encoder Cross attention proposed in [2] with the model proposed in [1]. There are 4 layers and in each layer 3 attention blocks are used. Each attention block has 4 attention heads. The two blocks take their key, query and values and inputs. Thirds block takes output of block1 as Query and Values and output of block 2 as Key. This Propose model showed better results as shown in Results Section.</p>
<div style="display: flex;">
<img src="https://github.com/ashutoshc8101/dl-minor-1/blob/main/images/architecture.png" alt="image" width="400" height="400"> <img src="https://github.com/ashutoshc8101/dl-minor-1/blob/main/images/cross_attention.png" alt="image1" width="400" height="400">
</div>
<h2 id="Results">Results</h2>
<p>The proposed architecture with Dual encoder cross attention has been trained on the concatenated features of MFCC + MelSpectogram + Chroma CQT. The results are stored in table 1.</p>
<p>Table 1: Comparison of performance of proposed model against the model in [1] which performs better than other quality techniques</p>
<Table>
   <tr>
      <th>Metric</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
   </tr>
   <tr>
      <th>Proposed Model</th>
      <th><B>0.828</B></th>
      <th><B>0.823</B></th>
      <th><B>0.629</B></th>
   </tr>
   <tr>
      <th>Model propose in [1]</th>
      <th>0.816</th>
      <th>0.812</th>
      <th>0.613</th>
   </tr>
   <tr>
      <th>Proposed model with 1 attention head in cross attention block</th>
      <th>0.823</th>
      <th>0.821</th>
      <th>0.619</th>
   </tr>
</Table>

<div style="display: flex;">
<img src="https://github.com/ashutoshc8101/dl-minor-1/blob/main/images/Epochs.png" alt="image" width="700" height="500"> 
</div>
<h2 id="ablation-study">Ablation Study</h2>

<h3> Table 2: Ablation Study of model proposed in [1] on different features individually.</h3>
<table>
    <tr>
        <th>Features</th>
        <th>PLCC</th>
        <th>SRCC</th>
        <th>KRCC</th>
    </tr>
    <tr>
        <td>MFCC</td>
        <td>0.642</td>
        <td>0.623</td>
        <td>0.449</td>
    </tr>
    <tr>
        <td>MelSpectogram</td>
        <td>0.578</td>
        <td>0.566</td>
        <td>0.400</td>
    </tr>
    <tr>
        <td>Chroma CQT</td>
        <td>0.321</td>
        <td>0.345</td>
        <td>0.241</td>
    </tr>
    <tr>
        <td>SPectral Contrast</td>
        <td>0.227</td>
        <td>0.207</td>
        <td>0.141</td>
    </tr>
</table>
<p>To study the contribution of different features we trained the model proposed in [1] on individual features. The correlation between the predicted output of the trained model and actual values is stored in Table21. It shows the best features are in order MFCC > Melspectogram > Chroma CQT > SPectral COntrast. Other than these none of the features(PNCC, Spectral Centroid) showed promising results.</p>
<br>
<h3>Table 3: Ablation Study of model proposed in [1] on different combination of features.</h3>
<table>
    <tr>
        <th>Experiments</th>
        <th>PLCC</th>
        <th>SRCC</th>
        <th>KRCC</th>
    </tr>
    <tr>
        <td>MFCC + MelSpectogram + Chroma CQT</td>
        <td><B>0.769</B</td>
        <td><B>0.763</B></td>
        <td><B>0.570</B></td>
    </tr>
    <tr>
        <td>MFCC + MelSpectogram + Spectral Contrast</td>
        <td>0.747</td>
        <td>0.736</td>
        <td>0.5430</td>
    </tr>
    <tr>
        <td>MFCC + Melspectogram + Chroma CQT + Spectral Contrast</td>
        <td>0.730</td>
        <td>0.726</td>
        <td>0.538</td>
    </tr>
    <tr>
        <td>MFCC + Melspectogram + Chroma CQT + SPectral Contrast + PNCC</td>
        <td>0.721</td>
        <td>0.716</td>
        <td>0.530</td>
    </tr>
   <tr>
        <td>MFCC + Melspectogram + Chroma CQT + PNCC</td>
        <td>0.297</td>
        <td>0.445</td>
        <td>0.305</td>
    </tr>
</table>
<p>As individual contribution is not enough to arrive at a conclusion, we also studied the performance of the model on the input of different combinations of features. To study this we trained the model proposed in [1] on the combinations shown in table 3. The correlation between the predicted output of the trained model and actual values is also shown in Table 2. As the dataset used has 2075 audio samples, concatenating too many features or features having large dimensions results in degradation of results due to <B>Curse of Dimensionality</B>.</p>
<p>We propose that in case of larger dataset, which can also be made using data augmentation, MFCC + Melspectogram + Chroma CQT + Spectral Contrast should be used but for this study we used MFCC + Melspectogram + Chroma CQT</p>

<h2 id="dependencies">Dependencies</h2>

<p>To install the required dependencies, simply run the following command:</p>

```bash
pip install -r requirements.txt
```
<p>Please ensure that you have these libraries installed to run the project.</p>

<h2 id="usage">Usage</h2>

<p>To use this project, follow these steps:</p>
<ol>
    <li>Clone this GitHub repository.</li>
    <li>Install the required dependencies.</li>
    <li>Train the model on your audio quality assessment dataset.</li>
    <li>Evaluate the model's performance.</li>
</ol>

<h2 id="acknowledgment">Acknowledgment</h2>

<p>We would like to acknowledge the support and contributions of the open-source community in making this project possible. Additionally, we extend our gratitude to the following researchers and their papers:</p>

<p><strong>1. Transformer-based quality assessment model for generalized user-generated multimedia audio content</strong></p>
<p>Dataset Credits: The dataset used in this project was generously provided by Mumtaz, D., Jena, A., Jakhetiya, V., Nathwani, K., and Guntuku, S.C. as described in their paper, "Transformer-based quality assessment model for generalized user-generated multimedia audio content" (Proc. Interspeech 2022, 674-678, doi: 10.21437/Interspeech.2022-10386).</p>

<p><strong>2. Improved Transformer Model for Enhanced Monthly Streamflow Predictions of the Yangtze River</strong></p>
<p>We acknowledge the work of C. Liu, D. Liu, and L. Mu as described in their paper, "Improved Transformer Model for Enhanced Monthly Streamflow Predictions of the Yangtze River" (IEEE Access, vol. 10, pp. 58240-58253, 2022, doi: 10.1109/ACCESS.2022.3178521).</p>

<p>We appreciate the valuable contributions of these researchers and the resources they provided for our project.</p>


<h2 id="team-members">Team Members</h2>
<p>This project was made possible by the efforts of our team members:</p>
<ul>
    <li>Ashutosh Chauhan</li>
    <li>Dakshi Goel</li>
    <li>Aman Kumar</li>
   <li>Devin Chugh</li>
   <li>Shreya Jain</li>
</ul>

<h2 id="contributing">Contributing</h2>

<p>We welcome contributions to enhance this project. If you would like to contribute, please follow the standard GitHub pull request process.</p>

<p>For any questions or issues, please open a GitHub issue in this repository.</p>

<p>Thank you for your interest in our audio quality assessment project!</p>


</body>

</html>
