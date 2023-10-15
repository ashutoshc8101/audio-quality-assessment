<!DOCTYPE html>
<html>

<head>
   
</head>

<body>

<h1>Audio Quality Assessment with Transformer-Based Learning</h1>

<p>This GitHub project introduces a novel approach to audio quality assessment using transformer-based deep learning architecture. The proposed model leverages the power of transformers to process audio data, providing enhanced performance over traditional approaches. This README provides an overview of the architecture, model configuration, and the tools used for this project.</p>



<h2>Architecture Overview and Model Configuration</h2>

<p>The proposed model employs a transformer-based deep learning approach to assess audio quality. It takes hand-crafted features concatenated into a vector as input and is trained with corresponding ground-truth labels. The transformer architecture, comprising an Encoder-Decoder structure with Multi-Head Attention (MHA) and Feed-Forward layers, processes the data. We utilized four layers of the encoder, set the number of heads (h) in each MHA to four, and employed an Adam optimizer. The model outputs a single continuous value representing audio quality in the range of 1 to 5. These design choices optimize feature vectors while considering attention mechanisms for enhanced performance.</p>

<p>
   The following audio extraction features are used:
   <br />
   
   <ol>
   <li><b>Melspectogram:</b> <br />
      Mel-Spectrogram is computed by applying a Fourier transform to analyze the frequency content of a signal and to convert it to the mel-scale, while MFCCs are calculated with a discrete cosine transform (DCT) into a melfrequency spectrogram.
   </li>
   <li>
      <b>MFCC:</b> <br />
      The mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10-20) which concisely describe the overall shape of a spectral envelope. In MIR, it is often used to describe timbre.
   </li>

   <li>
      <b>Spectral Contrast:</b> <br />
      Spectral contrast is defined as the decibel difference between peaks and valleys in the spectrum.
   </li>

   <li>
      <b>PNCC:</b> <br />
      PNCCs introduce additional signal enhancement and noise compensation operations on the spectrogram.
   </li>
   </ol>
</p>

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
