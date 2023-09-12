# Unsupervised EEG Artifact Detection and Correction Summary

Created by: Imagine Software
Created time: September 12, 2023 2:12 PM
Authors: https://www.frontiersin.org/people/u/11556191,2*  Eric Chantland2  Tuka Alhanai3  https://www.frontiersin.org/people/u/74432 https://www.frontiersin.org/people/u/9604881

You can find the full published paper here: [https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full)

## Summary

EEG is vital for diagnosing various neurological disorders but faces challenges from various noise sources. Current artifact detection methods demand manual input, while correction techniques adopt a "one-size-fits-all" approach. This paper introduces a new EEG noise-reduction technique using representation learning for specific artifact detection and correction. The method identifies EEG artifacts unique to tasks and subjects, then uses a deep network for artifact correction. Tests showed a 10% performance boost with this method. This unsupervised framework doesn't need expert oversight and suits various clinical tasks. The method, code, and data are publicly available, offering practical utility and a basis for future research.

### Introduction

EEG devices, used in various fields, face challenges from multiple noise sources like movement artifacts and physiological disturbances. Many artifact identification methods necessitate manual labeling, specialized hardware, or vast template data-sets. Manually annotating artifacts is inefficient, especially when artifacts vary depending on tasks, subjects, or experimental trials. Simple removal of artifacts can interfere with data analyses. The definition of "artifact" is task-dependent, making unsupervised, data-driven approaches ideal for detecting them. Unsupervised methods work best when artifacts are rare in specific tasks. While current techniques excel at detecting specific artifact types, there's a need for more general approaches.

Artifact removal, a broader noise reduction challenge, has been explored since the 1940s. One prevalent EEG artifact removal technique is Independent Component Analysis (ICA). However, ICA requires expert review and has limitations, especially with fewer channels. One common artifact, the electrode "pop", results from sudden impedance changes. While interpolation can address this, precise electrode locations often improve results. Recently, deep learning convolutional auto-encoders were introduced for subject-specific interpolation without human annotation.

This paper presents a comprehensive approach for artifact detection and removal without task or artifact type assumptions. The method employs various EEG features for tasks like coma prognosis and mental illness diagnosis. Unsupervised outlier detection algorithms pinpoint artifacts in EEG data. For artifact correction, a deep encoder-decoder network is used, framing the task as "frame-interpolation". This novel approach doesn't rely on extensive template data-sets. The paper delves into the datasets used, model architecture, experimentation, and results, concluding with findings, implications, and limitations.

### Methods

This paper introduces an automated EEG pre-processing pipeline for detecting, rejecting, and correcting artifacts using both feature-based and deep-learning techniques. It's intended as a versatile EEG pre-processing tool. We start with a summary of the data and methods, directing readers to specific sections for comprehensive details.

Figure 1 visualizes our pipeline. We begin with unsupervised detection of epoched EEG segments in a 58-dimensional feature space (subsection 2.2). Trials passing this stage train a deep encoder-decoder network to rectify artifact segments (subsection 2.3).

![****************Figure 1****************](Unsupervised%20EEG%20Artifact%20Detection%20and%20Correction%20b8592b06703f4a0fb4a054ef0731ae4d/Untitled.png)

****************Figure 1****************

We present an adaptable EEG pre-processing technique suitable for any EEG dataset. This paper arranges the methods based on their order in the pipeline.

**2.1. Data-Sets**

**2.1.1. Data Acquisition**

Our objective is to demonstrate the effective use of unsupervised anomaly detection in EEG data and its subsequent correction through representation learning. For a robust assessment, we require datasets with both artifact annotations and trial labels. However, most datasets lack this comprehensive labeling. Hence, we use two datasets, the orientation and color datasets, collected by Saidya et al. (20).

These datasets come from passive viewing tasks involving oriented gratings and random dot fields in various colors. Data was captured from a 32-electrode actiCHamp cap at 1,000 Hz, involving seven subjects, resulting in around 10,000 EEG Trials. After expert review for noise, the data will be available online. All participants provided informed consent, were compensated, and the study complied with Michigan State University's ethical guidelines.

**2.2. Unsupervised Artifact Detection**

We aimed to assess the potential of unsupervised artifact rejection in EEG by comparing different outlier detection methods. This involved gathering commonly-used EEG features and applying various unsupervised outlier detection algorithms.

**2.2.1. Feature Extraction**

Building on Ghassemi et al. (21), we identified 58 commonly-used features from EEG literature. The feature extraction process was optimized for parallel computation, and the code is available as a Python 3.5 package. A detailed list of these features is provided in Table 1.

****************Table 1.****************

![Untitled](Unsupervised%20EEG%20Artifact%20Detection%20and%20Correction%20b8592b06703f4a0fb4a054ef0731ae4d/Untitled%201.png)

The identified features fall into three main categories: complexity, continuity, and connectivity of EEG activity. Each category provides distinct insights:

- **Complexity**: Measures the intricacy of EEG patterns, revealing the brain's dynamic activities and processes.
- **Continuity**: Assesses the consistency and flow of EEG signals over time, indicating how stable or erratic brain activities might be.
- **Connectivity**: Evaluates the interrelationship and synchronization between different EEG channels, offering insights into how different brain regions interact.

For a deeper dive into the specifics of these features, we recommend reviewing Ghassemi et al.'s work (21).

**2.2.1.1. Complexity Features (n = 25)**

Complexity features assess EEG signal intricacy using information-theoretic measures. They correlate with cognitive impairments and degenerative diseases. Notably, Shannon's entropy relates to outcomes in post-anoxic coma patients, while the entropy of decomposed EEG wavelet signals and Tsallis entropy provide value in EEG complexity characterization.

**2.2.1.2. Continuity Features (n = 27)**

These gauge the regularity and fluctuations in EEG activity. Relevant features include bursts, spikes, and unusual frequency shifts, vital for clinical applications. For a thorough understanding, refer to **Hirsh et al.**

**2.2.1.3. Connectivity Features (n = 6)**

These capture EEG signal interdependencies across channels, reflecting functional brain connectivity. We extract features from extensive EEG connectivity literature, beneficial for various research areas.

**2.2.2. Outlier Detection Methods**

We assessed ten unsupervised artifact detection algorithms, inspired by Zhao et al., categorized as statistical methods and representation learning methods.

**2.2.2.1. Statistical Methods**

Statistical approaches identify outliers using data's statistical properties, producing an "anomaly score". We explored methods like Histogram-Based Outlier detection (HBOS), Local Outlier Factor (LOF), Angle-Based Outlier Detector (ABOD), and One Class SVM Detector (OCSVM). Additionally, we trained ensemble classifiers, specifically the Locally Selective Combination in Parallel (LSCP) Outlier Ensembles.

**2.2.2.2. Representation Learning Based Methods**

These methods, unlike statistical ones, delve deeper than just statistical properties. Auto-encoder (AUTO) based classifiers learn data's lower-dimensional representation, ensuring optimal original signal reconstruction. Variational Auto-Encoders (VAE) and Generative Adversarial Active Learning (GAAL) outlier detectors offer more sophisticated approaches.

**2.3. Artifact Correction**

Building on encoder-decoder deep learning models, this section presents an extension for artifact correction. It's akin to the "frame-interpolation" task in video processing.

**2.3.1. The Model**

**2.3.1.1. Input Representation**

Saba-Sadiya et al.'s model depicted EEG as time series of 2D arrays. In our model, given the importance of temporal modulations in EEG, we represent input as a 2D array, focusing on time evolution.

**2.3.1.2. Architecture**

Effective frame interpolation models consider object trajectories. In EEG, with its singular spatial dimension, we focus on global modulations. Thus, we employ a stacked convolutional auto-encoder, similar to frame and channel interpolation models. To predict multiple frames, we train separate networks for parallelized training. Given an EEG frame series, our network predicts missing frames based on surrounding frames. Each network predicts one specific frame, drawing from the same adjacent frames to compute the value at a given time.

**2.4. Model Validation Approach**

**2.4.1. Artifact Detection Method**

We gauged the efficacy of our artifact detection techniques by comparing them with expert annotations from the color and orientation datasets. The f-score and Cohen's Kappa were used to measure the agreement. We benchmarked our model's performance against an anticipated classifier that precisely knows the artifact count, which is expected to yield an f-score of 0.172 and a Kappa of 0.029. We executed the detection algorithms for individual subjects and the combined data. We anticipate a performance dip with the aggregated configuration due to the unique artifacts each EEG recording might introduce, stemming from varying setups and subject-specific factors.

**2.4.2. Artifact Correction Method**

For the artifact correction model's parameter optimization, we sourced training data from artifact-free trials, determined by our unsupervised artifact detection method. We randomly deleted a segment from the trial's center, using the h samples before and after the removed segment as the model's input and the deleted segment as the ground truth. The hyper-parameter h was optimized on the training set. All EEG data were re-sampled to 200Hz for validation, with reconstructed segments lasting 200ms each.

**2.4.3. End-to-end Assessment Approach**

To ascertain if our artifact correction method could improve downstream EEG tasks, we conducted various tests. Specifically, we trained two SVM models on the color dataset: one with raw data and the other post-artifact correction. Both models underwent 5-fold cross-validation, with their performance (μ and σ) assessed on the test set.

Furthermore, we evaluated our artifact correction method's impact on clean EEG trials to check if our method inadvertently deteriorated the quality of clean segments. We applied our artifact correction to 20% of the clean trials and trained an additional SVM model using the modified data.

**3. Results**

This section delves into the outcomes from our pipeline's two primary components - the artifact detection and correction methods, based on the data detailed in section 2.1.

**3.1. Artifact Detection Results**

Table 2 showcases the average performance of the outlier detection techniques discussed in section 2.2.2, evaluated on individual subjects. Each presented value represents the algorithm's average performance across all subjects. As indicated earlier, a baseline random classifier, aware of the precise artifact count, is expected to yield an f-score of 0.172 and a Kappa of 0.029. With this benchmark, every model except the ABOD classifier surpassed the baseline performance significantly (verified using a one-tailed t-test at a p=0.05 significance threshold).

******************Table 2******************

![Untitled](Unsupervised%20EEG%20Artifact%20Detection%20and%20Correction%20b8592b06703f4a0fb4a054ef0731ae4d/Untitled%202.png)

As observed, the standout outlier detector was the LSCP ensemble classifier, which exhibited a performance 16.86 times superior to the baseline and 1.03 times superior to the subsequent best method. The optimal configuration for this classifier comprised two HBOS classifiers paired with one OCSVM. It's interesting to note the contrast in the two histogram-based classifiers: one utilized a large number of histogram bins with a strict outlier scoring criterion (tol = 0.1), while the other adopted fewer bins with a more lenient policy (tol = 0.5). The best-performing representation learning algorithm was a simple auto-encoder, closely trailed by the PCA algorithm. We postulate that the auto-encoder's performance might have further improved with increased data availability per subject. For a detailed breakdown of trials and artifact counts per subject, refer to our Supplementary Material.

Table 3 presents the results when applying the outlier detection methods from section 2.2.2 to aggregated data across subjects, contrasting the subject-specific approach of Table 2. A noticeable performance dip is evident for most models in the aggregated setting. This decline aligns with expectations, as unsupervised methods inherently presume data homogeneity, with outliers being the exceptions. Yet, the LSCP method remained the top-performer. Comparing results from Tables 2 and 3 underscores the importance of tailoring anomaly detection to specific subjects. Furthermore, it accentuates the capability of our chosen unsupervised algorithms and extracted features to effectively discern both general EEG artifacts and individual-specific quirks.

****************Table 3.****************

![Untitled](Unsupervised%20EEG%20Artifact%20Detection%20and%20Correction%20b8592b06703f4a0fb4a054ef0731ae4d/Untitled%203.png)

**3.2. Artifact Correction Results**

**3.2.1. Network Optimization**

Initially, we focused on fine-tuning the network's hyper-parameters. This involved experimenting with varying layer and convolution filter sizes, along with other hyper-parameters like optimization strategies, dropout rates, and activation functions. To train the model, we adhered to the methodology detailed in section 2.2.2. We randomly selected 104 samples from the data; the leading and trailing 32 samples formed the input to our model, while the ith sample from the remaining 40 served as the ground truth. The idea was to train the network to predict the values after a 200ms (or 40 samples) gap, using the adjacent 32 samples on either side. Interestingly, the top-performing network (based on the lowest loss) varied depending on the specific time step, \( t \). The optimal configuration for reconstructing the 20th sample can be found in the Supplementary Material, exemplifying the convolutional U-net architecture we employed.

**3.2.2. End-to-End Assessment**

Table 4 presents a comparison of classification accuracies achieved by a 5-fold SVM model tasked with downstream trial type classification using down-sampled EEG data in three distinct configurations:

1. The unaltered, raw EEG data.
2. The data post artifact segment correction.
3. The data following a "correction" of a random 40-sample segment in 20% of the artifact-free trials.

It's pertinent to highlight that, despite its simplicity, such a comparative analysis is indeed a staple in genuine EEG research endeavors, as noted in reference (4).

****************Table 4.****************

![Untitled](Unsupervised%20EEG%20Artifact%20Detection%20and%20Correction%20b8592b06703f4a0fb4a054ef0731ae4d/Untitled%204.png)

Using the artifact correction method on artifact-free trials resulted in nearly equivalent performance to the original. This finding suggests that our model effectively learns to reconstruct the inherent EEG signal without introducing significant distortions. More notably, when we applied the correction to trials with EEG artifacts, there was a substantial improvement in classification accuracy—increasing by 10% overall. Moreover, for the trials specifically flagged for containing artifacts, the accuracy boost exceeded 20%. Such outcomes compellingly showcase that our unsupervised, end-to-end artifact correction pipeline can significantly enhance the quality of downstream analyses. This has profound implications for EEG research, suggesting that even when faced with noisy or artifact-laden data, robust preprocessing can lead to accurate and meaningful results.

**4. Discussion**

**4.1. Significance of Our Results**

The current research showcases the capabilities of an end-to-end pipeline designed for unsupervised EEG artifact detection and correction. The implications of this study suggest that leveraging data-driven techniques for unsupervised outlier detection is particularly potent when addressing EEG artifact detection. The standout performance of global classifiers like HBOS, OCSVM, and LSCP suggests a possible inclination of EEG artifacts being better distinguished by global traits. This revelation aligns with our earlier insights that artifacts in EEG are task-centric and are sporadic instances of uncorrelated disturbances. Notably, these classifiers, as evidenced in Table 3, have exhibited competence in discerning subject-specific peculiarities.

Despite the less than ideal accuracy and agreement between annotators and detectors, the Cohen Kappa metric for the top-performing algorithm mirrors the inter-rater agreement levels reported in other studies. For context, expert annotators yielded a Cohen's Kappa of 0.38 and 0.58 when annotating specific artifact types (51). This underscores the potential and feasibility of unsupervised outlier detection in general EEG artifact identification.

**4.2. The Data-Sets**

Our methodology was assessed using two unique data-sets. To gauge the efficacy of artifact correction algorithms, it's crucial to possess both an authoritative artifact annotation and a comprehensive label set for all trials, artifacts included. Regrettably, publicly available data-sets often omit artifact-containing trials. We aspire for our data-sets to encourage researchers to adopt comprehensive data publication practices, as limited data availability remains a significant hurdle in artifact correction research.

**4.3. The Strength of Unsupervised End-to-End Methods**

Following artifact removal, the precision of rudimentary classifiers witnessed a modest enhancement. Substituting our deep learning artifact removal techniques with an ICA algorithm could potentially yield superior outcomes. However, it's imperative to distinguish two points: our proposed model avoids certain ICA-related pitfalls, and while ICA is unsupervised and data-driven, it demands expert human scrutiny of the decomposed signal. Conversely, our proposed method operates independently of human intervention, making it apt for real-time EEG applications. While supervised techniques undoubtedly overshadow unsupervised ones, our pipeline is no exception. Hence, unsupervised methods are better seen as complementary tools rather than replacements. With this philosophy, we crafted our pipelines to be adaptable, allowing seamless integration or substitution with other methods, tailored to researchers' needs.

**4.4. Limitations**

We abstained from a formal assessment of the model's reconstruction performance due to the absence of a universally recognized benchmark in literature. However, if the reconstruction augments the performance of subsequent classification models, it stands validated. Several limitations need addressing: the artifact detection method's reliance on the infrequency of artifacts, potential enhancement of our artifact correction network by introducing intricate components, and the pressing need for validation on diverse tasks and data-sets.

Nonetheless, we are optimistic about our research, believing it embodies a viable EEG preprocessing framework that, if embraced, could streamline the often intricate process of artifact annotation and removal, offering considerable value to the broader EEG research fraternity.

**5. Conclusion and Future Work**

EEG applications are vast and varied, and while the specifics of what constitutes signal vs. artifact might differ, the challenge of data uniformity persists. Approaching this from a data science viewpoint, our study integrates advanced data-driven methods to devise a comprehensive unsupervised pipeline for general artifact detection and correction. By introducing two novel data-sets, we've shown that our artifact detection tool's inter-rater reliability against expert annotations aligns with known human-to-human levels. We also exemplify how utilizing the complete pipeline on a data-set can refine the outcomes of prevalent downstream analysis. Our pipeline leverages an extensive array of meticulously crafted, clinically pertinent features, and we are confident that the Python package we've released will find resonance within the EEG research community.

## 4. Discussion

### 4.1. Significance of Our Results

In this paper, we presented an end-to-end pipeline that 
is capable of unsupervised artifact detection and correction. Our 
results demonstrate that data-driven approaches for unsupervised outlier
 detection can be extremely useful when applied to the problem of EEG 
artifact detection. Interestingly, the classifiers with the best 
performance (HBOS, OCSVM, and the best performing LSCP) are global 
classifiers; this might indicate that EEG artifacts are better 
discriminated by global characteristics. This supports our previous 
observation that artifacts are task specific and infrequent occurrences 
of uncorrelated noise. It is worth noting that, as demonstrated in [Table 3](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#T3), the classifiers we trained were able to learn subject-specific idiosyncrasies.

While the accuracy and agreement between the annotators 
and the detectors were far from perfect, the Cohen Kappa of the best 
performing algorithm was comparable to the inter-rater agreement levels 
of expert annotators reported in the literature; for instance, when 
asked to annotate, “periodic discharges” (a specific type of artifact) 
and “electrographic seizure” annotators had a Cohen's Kappa of 0.38 and 
0.58, respectively ([51](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#B51)). Our results indicate that an unsupervised outlier detection is a feasible approach for generalized EEG artifact detection.

### 4.2. The Data-Sets

We validated our framework on two novel data-sets. To 
test the impact of artifact correction algorithms on downstream analysis
 it is necessary to have ground truth artifact annotation as well as 
knowledge of the labels of all trials, including those that are artifact
 ridden. Unfortunately, public data-sets often exclude trials that 
contain artifacts. Even in the rare occasions in which these trials are 
made available, the labels are often replaced with a special identifier 
for rejected trials[4](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#note4).
 We hope our data-sets inspire other researchers to adopt more thorough 
data publishing practices as data-availability is perhaps the primary 
limiting factor in artifact correction research.

### 4.3. The Strength of Unsupervised End-to-End Methods

The accuracy of simple classifiers improved modestly 
after artifact removal. It is possible that replacing our 
deep-learning-based artifact removal components with an ICA artifact 
removal algorithm ([52](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#B52))
 could yield better results. However, two important distinctions should 
be made: First, the proposed method does sidestep many weaknesses 
inherent to ICA ([8](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#B8))
 (such as the number of independent components being limiting by the 
number of channels, which is particularly problematic for lightweight 
commercial EEG setups). Secondly, while the independent component 
deconstruction itself is data driven and unsupervised, the ICA method 
still requires visual inspection and analysis of the decomposed signal 
by human experts. In contrast, our method can be put into effect without
 any human intervention, making it is suitable for online EEG 
applications or as a no-cost first step before a more thorough analysis.
 In general, supervised methods unquestionably out-perform unsupervised 
ones and we fully acknowledge that the pipeline proposed in this work is
 no different. It is therefore useful to consider unsupervised methods 
not as replacements of currently existing algorithms but as 
complimentary additions to the toolbox of the EEG researcher. With this 
in mind, we intentionally designed our end-to-end pipelines to be highly
 modular; An experienced researcher can easily substitute our last 
component with an ICA artifact removal algorithm, and in contrast, 
researchers that have access to artifact annotations (for instance by 
virtue of employing specialized hardware during data acquisition) will 
be able to use their method in conjunction with ours or sidestep the 
first processes completely and apply only the artifact correction 
component before carrying on with the analysis process.

### 4.4. Limitations

We did not formally evaluate the reconstruction 
performance of the model because (1) there is not an authoritative 
literature baseline, and (2), insofar as the reconstruction enhances the
 ability of the downstream classification model to perform their 
intended classification tasks, the reconstruction is valid and valuable.
 There are a few limitations that we hope to address in future work. 
First and foremost, this artifact detection method can only be used if 
the frequency of the artifacts is low enough for them to be considered 
outliers. While this is indeed the case for the vast majority of EEG use
 cases, tasks, such as seizure detection often involve long periods of 
unusually low signal to noise ratio. Additionally, the performance of 
our artifact correction network would likely benefit from introducing 
more complex component into the architecture. For instance, introducing 
temporal dependencies via an LSTM component would guarantee that the 
corrected frame at time *t* influences the frame at time *t*+1. Finally, our method is in dire need of being validated on additional tasks and data-sets.

Despite the challenges described above, we believe that 
our work demonstrates the feasibility of an EEG pre-processing pipeline 
which if adopted could facilitate and expedite the often tenuous process
 of artifact annotation and removal, and could therefore be extremely 
beneficial for the general EEG research community.

## 5. Conclusion and Future Work

The applications of EEG are numerous and diverse, and 
while this impacts the particularities of what components are classified
 as part of the signal vs. artifacts, data homogeneity is a common 
concern in this area of research. Building on this data science 
perspective, in this work we appropriated state-of-the-art data-driven 
methods to construct an end-to-end unsupervised pipeline for general 
artifact detection and correction. We introduced two new data-sets and 
demonstrated that the inter-rater reliability of our artifact detection 
component against expert annotators is comparable to reported 
inter-human levels. Furthermore, we demonstrated how applying the 
complete pipeline on a data-set can improve the performance of common 
downstream analysis. The pipeline makes use of a wide range of 
handcrafted clinically relevant features, and we believe the released 
python package will be of use to many in the EEG research community.

## 4. Discussion

### 4.1. Significance of Our Results

In this paper, we presented an end-to-end pipeline that 
is capable of unsupervised artifact detection and correction. Our 
results demonstrate that data-driven approaches for unsupervised outlier
 detection can be extremely useful when applied to the problem of EEG 
artifact detection. Interestingly, the classifiers with the best 
performance (HBOS, OCSVM, and the best performing LSCP) are global 
classifiers; this might indicate that EEG artifacts are better 
discriminated by global characteristics. This supports our previous 
observation that artifacts are task specific and infrequent occurrences 
of uncorrelated noise. It is worth noting that, as demonstrated in [Table 3](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#T3), the classifiers we trained were able to learn subject-specific idiosyncrasies.

While the accuracy and agreement between the annotators 
and the detectors were far from perfect, the Cohen Kappa of the best 
performing algorithm was comparable to the inter-rater agreement levels 
of expert annotators reported in the literature; for instance, when 
asked to annotate, “periodic discharges” (a specific type of artifact) 
and “electrographic seizure” annotators had a Cohen's Kappa of 0.38 and 
0.58, respectively ([51](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#B51)). Our results indicate that an unsupervised outlier detection is a feasible approach for generalized EEG artifact detection.

### 4.2. The Data-Sets

We validated our framework on two novel data-sets. To 
test the impact of artifact correction algorithms on downstream analysis
 it is necessary to have ground truth artifact annotation as well as 
knowledge of the labels of all trials, including those that are artifact
 ridden. Unfortunately, public data-sets often exclude trials that 
contain artifacts. Even in the rare occasions in which these trials are 
made available, the labels are often replaced with a special identifier 
for rejected trials[4](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#note4).
 We hope our data-sets inspire other researchers to adopt more thorough 
data publishing practices as data-availability is perhaps the primary 
limiting factor in artifact correction research.

### 4.3. The Strength of Unsupervised End-to-End Methods

The accuracy of simple classifiers improved modestly 
after artifact removal. It is possible that replacing our 
deep-learning-based artifact removal components with an ICA artifact 
removal algorithm ([52](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#B52))
 could yield better results. However, two important distinctions should 
be made: First, the proposed method does sidestep many weaknesses 
inherent to ICA ([8](https://www.frontiersin.org/articles/10.3389/fdgth.2020.608920/full#B8))
 (such as the number of independent components being limiting by the 
number of channels, which is particularly problematic for lightweight 
commercial EEG setups). Secondly, while the independent component 
deconstruction itself is data driven and unsupervised, the ICA method 
still requires visual inspection and analysis of the decomposed signal 
by human experts. In contrast, our method can be put into effect without
 any human intervention, making it is suitable for online EEG 
applications or as a no-cost first step before a more thorough analysis.
 In general, supervised methods unquestionably out-perform unsupervised 
ones and we fully acknowledge that the pipeline proposed in this work is
 no different. It is therefore useful to consider unsupervised methods 
not as replacements of currently existing algorithms but as 
complimentary additions to the toolbox of the EEG researcher. With this 
in mind, we intentionally designed our end-to-end pipelines to be highly
 modular; An experienced researcher can easily substitute our last 
component with an ICA artifact removal algorithm, and in contrast, 
researchers that have access to artifact annotations (for instance by 
virtue of employing specialized hardware during data acquisition) will 
be able to use their method in conjunction with ours or sidestep the 
first processes completely and apply only the artifact correction 
component before carrying on with the analysis process.

### 4.4. Limitations

We did not formally evaluate the reconstruction 
performance of the model because (1) there is not an authoritative 
literature baseline, and (2), insofar as the reconstruction enhances the
 ability of the downstream classification model to perform their 
intended classification tasks, the reconstruction is valid and valuable.
 There are a few limitations that we hope to address in future work. 
First and foremost, this artifact detection method can only be used if 
the frequency of the artifacts is low enough for them to be considered 
outliers. While this is indeed the case for the vast majority of EEG use
 cases, tasks, such as seizure detection often involve long periods of 
unusually low signal to noise ratio. Additionally, the performance of 
our artifact correction network would likely benefit from introducing 
more complex component into the architecture. For instance, introducing 
temporal dependencies via an LSTM component would guarantee that the 
corrected frame at time *t* influences the frame at time *t*+1. Finally, our method is in dire need of being validated on additional tasks and data-sets.

Despite the challenges described above, we believe that 
our work demonstrates the feasibility of an EEG pre-processing pipeline 
which if adopted could facilitate and expedite the often tenuous process
 of artifact annotation and removal, and could therefore be extremely 
beneficial for the general EEG research community.

## 5. Conclusion and Future Work

The applications of EEG are numerous and diverse, and 
while this impacts the particularities of what components are classified
 as part of the signal vs. artifacts, data homogeneity is a common 
concern in this area of research. Building on this data science 
perspective, in this work we appropriated state-of-the-art data-driven 
methods to construct an end-to-end unsupervised pipeline for general 
artifact detection and correction. We introduced two new data-sets and 
demonstrated that the inter-rater reliability of our artifact detection 
component against expert annotators is comparable to reported 
inter-human levels. Furthermore, we demonstrated how applying the 
complete pipeline on a data-set can improve the performance of common 
downstream analysis. The pipeline makes use of a wide range of 
handcrafted clinically relevant features, and we believe the released 
python package will be of use to many in the EEG research community.