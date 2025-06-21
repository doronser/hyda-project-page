HyDA: Hypernetworks for Test Time Domain
Adaptation in Medical Imaging Analysis

Doron Serebro1 and Tammy Riklin-Raviv1

1 Ben-Gurion University of The Negev
2 doronser@post.bgu.ac.il
3 preprint version

Abstract. Medical imaging datasets often vary due to differences in ac-
quisition protocols, patient demographics, and imaging devices. These
variations in data distribution, known as domain shift, present a signifi-
cant challenge in adapting imaging analysis models for practical health-
care applications.
Most current domain adaptation (DA) approaches aim either to align
the distributions between the source and target domains or to learn an
invariant feature space that generalizes well across all domains. How-
ever, both strategies require access to a sufficient number of examples,
though not necessarily annotated, from the test domain during training.
This limitation hinders the widespread deployment of models in clinical
settings, where target domain data may only be accessible in real time.
In this work, we introduce HyDA, a novel hypernetwork framework that
leverages domain characteristics rather than suppressing them, enabling
dynamic adaptation at inference time. Specifically, HyDA learns implicit
domain representations and uses them to adjust model parameters on-
the-fly, effectively interpolating to unseen domains. We validate HyDA on
two clinically relevant applications—MRI brain age prediction and chest
X-ray pathology classification—demonstrating its ability to generalize
across tasks and modalities. Our code is available at TBD.

Keywords: Domain Adaptation · Hypernetworks · MRI · X-Ray

1

Introduction

Deep learning significantly advanced medical image analysis by enabling accurate
detection, classification, segmentation, and predictive modeling. However, for
practical healthcare applications, models must adapt to variations in imaging
protocols, scanner types, and patient demographics, which create differing data
distributions between training and test sets. This discrepancy, known as domain
shift, remains a significant barrier to robust model performance.

Current domain adaptation (DA) techniques generally aim to either align the
distributions between source and target domains or learn a consistent feature
space across different domains. Yet, both approaches depend on having enough
target domain samples during training, whether annotated or not. This depen-
dency poses a challenge for the deployment of models in clinical settings, where

2

D. Serebro et al.

the target data may only be available at the time of the test. In this work, we
propose HyDA, a novel hypernetwork framework that exploits domain charac-
teristics rather than discarding them, enabling dynamic adaptation during both
training and inference. Specifically, HyDA learns implicit domain representations
that are used to generate weights and biases for a primary network on-the-fly,
effectively interpolating to unseen domains. HyDa is task and modality agnostic,
making it easily integrable into various medical imaging applications. We show-
case its generality and robustness for two clinically relevant tasks—chest X-ray
pathology classification and MRI brain age prediction, demonstrating superior
performance over baseline and other domain adaptation techniques.

2 Related Works

Domain Adaptation Methods. Unsupervised domain adaptation (UDA) ad-
dresses shifting data distributions between source and target domains. One
prominent approach is domain adversarial learning, as exemplified by Domain-
Adversarial Neural Networks (DANN)[8], while MDAN (Multi-Domain Adver-
sarial Network) extends this idea to multiple source domains[33]. Another line
of works focuses on invariant feature learning; for example, Deep CORrelation
ALignment (CORAL) minimizes domain discrepancy by aligning the second-
order statistics of source and target features [27].
Recently, transformer-based methods have gained traction for their self-attention
capabilities. TransDA leverages domain-specific tokens and cross-attention to
align features unsupervisedly [31]. Similarly, AdaptFormer [3] and DAFormer [11]
integrate lightweight adapter modules within a Vision Transformer framework
to modulate representations based on domain cues while maintaining a shared
global representation.

Test-time domain adaptation (TTDA) techniques have a key advantage over
the methods mentioned above: they do not require target data during training
and can adapt models on-the-fly during inference. For example, TENT (Test
Time Entropy Minimization) adjusts model parameters via entropy minimiza-
tion, which works well for multi-class classification tasks with clear output prob-
abilities [28]. MEMO stabilizes adaptation under distribution shifts using aug-
mentations [32]. Although not strictly a TTDA method, SHOT (Source Hy-
pothesis Transfer) adapts to target data without requiring source samples [15]
by relying on pseudo-labeling and entropy minimization. However, the reliance
on entropy may limit their applicability to tasks such as regression or multi-label
classification without modifications.

Hypernetworks. First introduced by Ha et al.[9], hypernetworks are neural
networks that generate weights and biases for primary networks, dynamically
creating a unique set of parameters for each input. Their effectiveness has been
demonstrated in various tasks, including 3D shape reconstruction[16], federated
learning [26], and medical image segmentation [18]. Aharon et al.[1] showed that
hypernetworks can interpolate by conditioning an image denoising model on ex-
pected noise variance, while Duenias et al.[6] used them to condition medical

Hypernetworks for Test Time Domain Adaptation in Medical Imaging

3

Fig. 1: The proposed HyDA framework (left) is composed of a hypernetwork
h (right), a primary network P, and a domain classifier D. The hypernetwork
generates weights and biases to the primary’s head Phead based of the domain
feature vector XD provided by the domain encoder Denc. Other weights in the
system are internal and are learned via back-propagation using a regularized task
dependent LRT , classification LCE or/and multi-similarity LM Sim loss functions
as illustrated by the dashed arrows.

image analysis on tabular data. Building on these ideas, we show that hyper-
networks can be applied to medical imaging domain adaptation by generating
weights from domain features, effectively interpolating across the domain space.

3 Method

Our proposed HyDA framework, illustrated in Fig. 1, is composed of a primary
network P that could have any architecture addressing any medical imaging
analysis task; a hypernetwork h and a domain classifier D. Being trained on
datasets from different source domains - the classifier learns implicit domain
features that are mapped by h to sets of weights and biases. These parameters,
termed external parameters are transferred to a subset of layers in P.

Formally, let x ∈ Rd, denote a d−dimentional input image (d ∈ {2, 3}). We
define by Denc and Dhead the domain encoder and domain head, respectively,
which together compose the classifier D.

The classifier is trained to predict the domain of a training image x as follows:

yD = Dhead(Denc(x))

(1)

where yD is the label of a source domain. We assume the availability of at least
two source domains. If deployed separately, the trained domain encoder maps
any input x into a domain feature vector, i.e.,

xD = Denc(x)

(2)

where xD ∈ fD is the domain feature vector of x and fD denotes the domain
feature space. The primary network P which is trained to predict an output yP

4

D. Serebro et al.

for x can be formalized as follows:

yP = Phead(Penc(x), h(Denc(x))),

(3)

where, Penc denotes the internal layers in P which are trained through a stan-
dard back propagation process and h(Denc(x)) defines its external domain-aware
weights and biases - generated by the hypernetwork h.

We hypothesize that during inference, feature vectors of a target domain xD,
unseen by the domain encoder before, are well embedded within the domain
feature space, fD, and can be represented as linear combinations of training
domain features. We aim to optimize the hypernetwork such that the metric
capturing inter- and intra-domain relationships within fD is preserved in the
external primary network parameters. Once optimally converged, HyDA can
interpolate to new target domains at test time.

3.1 Domain Conditioning Hypernetwork

The hypernetwork maps the domain embedding xD to weights and biases (wh, bh)
for the external primary network layers. Let N, O, B denote the layer’s in-
put, output and batch size, respectively. In a standard linear layer, the output
ψ ∈ R(B,O) is computed as:

ψ = χ ∗ w, w ∈ R(N,O)

where χ ∈ R(B,N ) is the input batch and ∗ is matrix multiplication. A hyper-
linear layer instead assigns a unique weight matrix to each batch element:

ψi = χi ∗ wi

h, wi

h ∈ R(N,O)

i = 1, . . . B

The hypernetwork is flexible and can be implemented in various ways. For
simplicity, we use a single linear layer followed by a ReLU activation, which is
sufficient to generate effective domain-aware weights for the primary network.

To ensure stable convergence, we initialize the hypernetwork weights as in
Chang et al. [2] such that the input variance is preserved in the primary network.
We also regularize the weights using the l2 norm.

3.2 Loss Functions

The hypernetwork and the internal primary network layers are trained in an
end-to-end manner with a regularized loss function as follows:

LRT = Ltask + λBP ∥wBP ∥2 + λh ∥wh∥2

(4)

where, RT stands for regularized task, BP for backpropagation, Ltask is a task-
dependent loss (e.g. cross-entropy for classification, MSE for regression), wBP de-
note the union of the hypernetwork’s and the internal primary network’s weights,
wh are the exterrnal primary network weights, generated by the hypernetwork,

Hypernetworks for Test Time Domain Adaptation in Medical Imaging

5

and λBP , λh are their corresponding coefficients. The domain classifier is trained
using the following loss:

LD = LCE + αLM Sim + λD ∥wD∥2
where LCE is the cross-entropy loss, LM Sim is multi-similarity loss as in Wang
et. al. [30], wD are the domain network’s weights and α, λD are coefficients.
The supervised CE loss LCE aims to correctly classify the input into source
domains, while the contrastive, multi-similarity loss encourages the separation
of embedded domain feature vectors into different domain-aware clusters.

(5)

The multi-similarity loss (LM Sim ) also supports the hypernetwork training -
allowing it to maintain domain-specific representation of the weights and biases
it generates for the primary network.

4 Experiments and Results

We demonstrate the proposed HyDA framework on two medical imaging analysis
tasks - chest X-ray pathology classification, and MRI brain age prediction.

4.1 Chest X-ray Pathology Classification

We trained our model for multi-label classification on chest X-ray scans from
three publicly available datasets, comparing HyDA to a baseline with no adap-
tation, a UDA method (soft MDAN [33]), and a TTDA method (TENT [28]).
In both cases, the domain classifier was pre-trained for robust initialization.
The Data. We use the NIH [29], CheXpert [12], and VinDr [21] datasets, select-
ing five classes—Atelectasis, Cardiomegaly, Consolidation, Effusion, and Pneu-
mothorax—that are common across all three, resulting in a combined dataset of
90,570 X-ray scans.
Implementation Details. We fine-tuned a DenseNet121 model pre-trained
on ImageNet, replacing its input and output layers to process single-channel
images and output five classes, following prior work [24,4]. The domain classifier
is a simple CNN with four convolution blocks and a linear classification layer,
while the hypernetwork is a multi layer perceptron (MLP) that generates a set of
weights and biases for the primary network (DenseNet). Both baseline and HyDA
models were trained for 150 epochs using the AdamW optimizer (learning rate:
1e-3, weight decay: 0.05) with a cosine annealing scheduler (minimum learning
rate: 1e-6).
Results. Table 1 reports chest X-ray experiment results in terms of area under
curve (AUC). HyDA outperforms the baseline in both fully supervised and leave-
one-out settings. Notably, the improvement correlates with the separability of
domain features that were not seen in training (see Fig. 2); domains with well-
clustered features (CheXpert and VinDr) show larger gains compared to NIH.

Ablation Study. Table 2 presents an ablation study of the proposed loss func-
tions. The results highlight the contribution of each loss component to achieving
the best possible performance.

6

D. Serebro et al.

-

NIH

Avg.

Method

Target
Domain

Pathologies (AUC) ↑
Atel. Cardio. Cons. Eff. Pneu.
0.87
0.94
0.86
0.85
Baseline
0.86
0.86
0.88
0.94
MDAN
0.86 0.94 0.89
HyDA
0.87
0.77
0.86
0.76
Baseline 0.70
0.77
0.86
0.76
0.67
MDAN
0.81
0.64
0.61
TENT
0.67
0.88 0.79
0.75
0.68
HyDA
0.74
0.87
0.73
0.81
Baseline
0.72
0.84
0.71
0.77
MDAN
0.77
0.76
TENT
0.76
0.89
0.82 0.89 0.74
0.82
HyDA
0.91
0.88
0.85
0.60
Baseline
0.89
0.87
0.88
MDAN 0.68
0.80
0.51
TENT
0.86
0.74
0.93 0.89 0.92
0.66
HyDA

0.95
0.96
0.97
0.81
0.89
0.70
0.89
0.86
0.76
0.86
0.85
0.76
0.82
0.72
0.87
Table 1: Chest X-ray classification results measured by AUC. Pathologies ab-
breviations: Atel (Atelectasis), Cardio (Cardiomegaly), Cons (Consolidation),
Eff (Effusion), Pneu (Pneumothorax). Each group compares different models on
the same target domain. Best results in bold.

0.89
0.90
0.91
0.78
0.79
0.69
0.80
0.80
0.76
0.81
0.82
0.80
0.83
0.73
0.85

CheXpert

VinDr

LCE LD
✓
✓
✓

✓
✓

M Sim Lh

M Sim

NIH

CheXpert VinDr

0.72 (0.11)
0.76 (0.09)

0.81 (0.13)
0.83 (0.10)
✓ 0.80 (0.08) 0.82 (0.05) 0.85 (0.10)

0.79 (0.05)
0.81 (0.06)

Table 2: Ablation study of the loss terms. Each row represents an incremental
combination of loss terms, including domain classifier’s cross-entropy (CE) LCE
and multi-similarity (MSim) LD
M Sim loss functions as well as hypernetworks’
MSim loss Lh
M Sim. Average AUC results (std in brackets) of the target domain
for each of the three datasets are reported.

4.2 Brain Age Prediction

To further assess our method demonstrating its being task agnostic we evaluated
its performances for age prediction from brain MRI scans.
The Data. We used 19 brain MRI datasets containing 26,691 scans. The scans
were preprocessed using the workflow in Levakov et. at. [14].
Implementation Details. Our primary network follows Levakov et. al. [14]
3D CNN comprised of 4 convolution blocks followed by a 4-layered-MLP. The
domain classifier follows a similar architecture, with fewer parameters (refer to
our code for further details). The model was trained using AdamW optimizer

Hypernetworks for Test Time Domain Adaptation in Medical Imaging

7

(a) All domains

(b) w/o NIH

(c) w/o CheXpert

(d) w/o VinDr

Fig. 2: t-SNE projections of domain feature in fully supervised and leave-one-out
settings. The plots illustrate the embedding of previously unseen domains in the
learned domain feature space : (a) All domains training with respect to training
(b) w/o NIH (blue) (c) w/o CheXpert (orange) and (d) w/o VinDr (green).

Model CNP[23] NKI [22] ixi [10] Oasis [19] ABIDE [5] ADNI [13] AIBL [7] PPMI [20] Camcan [25] SLIM [17] Avg. (std)

Fully Supervised (Validation MAE) ↓

Baseline
HyDA

3.11
2.39

3.01
2.92

3.54
3.22

3.29
3.29

2.09
1.74

2.80
3.04

2.74
2.94

4.23
3.94

3.35
3.21

0.47
0.37

2.86 (0.96)
2.71 (0.95)

Leave-One-Out (Test MAE) ↓

3.90
3.44

3.36
2.86

Baseline
4.31
3.73 (0.97)
HyDA
4.48
3.57 (1.00)
Table 3: Brain age prediction results in fully supervised (validation MAE) and
leave-one-out (test MAE) settings. Best scores are in bold.

4.41
4.14

3.25
3.16

5.40
5.20

3.56
3.45

1.44
1.34

4.15
4.24

3.50
3.35

with a learning rate of 1e−4, weight decay of 0.05 and a cosine annealing learning
rate scheduler with a minimum learning rate of 1e − 6 for 150 epochs.
Results. Table 3 shows brain age prediction results. Notably, HyDA improves
over the baseline for both supervised a leave-one-out settings. These results
demonstrate HyDA’s ability to learn meaningful domain representations, inter-
polate to unseen domains and utilize the domain features to adapt the model
on-the-fly. The ability to interpolate is demonstrated in the t-SNE plot in Fig. 3
which shows how samples from a previously unseen domain are well embedded
in between feature vectors from domains used for training.
Ablation Study We evaluated the robustness of the HyDA by testing differ-
ent configurations of the primary network’s MLP head, which consists of four
layers, three of which can be external, having their weights generated by the
hypernetwork. Table 4 shows that replacing some internal layers with domain-
specific (external) ones improves performance over the baseline, regardless of
which layers are adapted. The best performance is achieved by combining both;
relying solely on task-specific weights limits generalization to unseen domains,
while using only domain-specific weights loses critical task-related information.

5 Conclusions

We introduced HyDA, a hypernetwork-based framework that rethinks test-time
domain adaptation in medical imaging by embracing domain variability rather

8

D. Serebro et al.

(a) All domains training

(b) All domains w/o Camcan

Fig. 3: t-SNE projections of domain feature vectors xD, Camcan examples are
in red. The plots show embedding when samples of all domains (a) or all but
Camcan (b) are available during training.

Layer 1 Layer 2 Layer 3 Average (std)

✓

✓

4.16 (0.26)
3.99 (0.20)
3.97 (0.32)
3.79 (0.21)
3.79 (0.35)
4.17 (0.11)

✓
✓
✓
Table 4: Hypernetwork external layer configuration - ablation study. Each con-
figuration was trained on two target domains (NKI, ixi), and results are reported
as mean (std) target domain MAE.

✓
✓

✓

than eliminating it. By learning implicit domain representations and dynamically
generating model parameters at test time, HyDA effectively tailors predictions
for each input based on its domain characteristics.

Experimental evaluations on chest X-ray pathology classification and MRI
brain age prediction demonstrate that HyDA outperforms traditional domain-
invariant methods and existing test-time adaptation techniques. The frame-
work’s ability to interpolate between domains, as revealed through t-SNE vi-
sualizations, confirms that leveraging domain-specific cues leads to more robust
and generalizable models. Moreover, the task-agnostic design and compatibility
with various architectures make HyDA a versatile solution for a wide range of
clinical applications. Overall, HyDA offers a promising pathway toward more
reliable and adaptable medical imaging analysis, paving the way for models that
can seamlessly adjust to real-world variations in data acquisition without the
need for extensive target domain training.

Hypernetworks for Test Time Domain Adaptation in Medical Imaging

9

References

1. Aharon, S., Ben-Artzi, G.: Hypernetwork-based adaptive image restoration. In:
International Conference on Acoustics, Speech and Signal Processing (ICASSP).
pp. 1–5 (2023)

2. Chang, O., Flokas, L., Lipson, H.: Principled weight initialization for hypernet-

works. In: International Conference on Learning Representations (2019)

3. Chen, S., Ge, C., Tong, Z., et al.: Adaptformer: Adapting vision transformers for
scalable visual recognition. Advances in Neural Information Processing Systems
(NeurIPS) 35, 16664–16678 (2022)

4. Cohen, J.P., Hashir, M., Brooks, R., Bertrand, H.: On the limits of cross-domain
generalization in automated x-ray prediction. In: Medical Imaging with Deep
Learning. pp. 136–155. PMLR (2020)

5. Di Martino, A., Yan, C.G., Li, Q., et al.: The autism brain imaging data exchange:
towards a large-scale evaluation of the intrinsic brain architecture in autism. Molec-
ular psychiatry 19(6), 659–667 (2014)

6. Duenias, D., Nichyporuk, B., Arbel, T., Riklin Raviv, T.: Hyperfusion: A hyper-
network approach to multimodal integration of tabular and medical imaging data
for predictive modeling. arXiv preprint arXiv:2403.13319 (2024)

7. Ellis, K.A., Bush, A.I., Darby, D., et al.: The australian imaging, biomarkers and
lifestyle (aibl) study of aging: methodology and baseline characteristics of 1112
individuals recruited for a longitudinal study of alzheimer’s disease. International
psychogeriatrics 21(4), 672–687 (2009)

8. Ganin, Y., Ustinova, E., Ajakan, H., et al.: Domain-adversarial training of neural

networks. Journal of machine learning research 17(59), 1–35 (2016)

9. Ha, D., Dai, A.M., Le, Q.V.: Hypernetworks. In: International Conference on

Learning Representations (2017)

10. Heckemann, R.A., Hartkens, T., Leung, K.K., et al.: Information extraction from
medical images: developing an e-science application based on the globus toolkit.
In: UK e-Science All Hands Meeting (2003)

11. Hoyer, L., Dai, D., Van Gool, L.: Daformer: Improving network architectures and
training strategies for domain-adaptive semantic segmentation. In: Conference on
Computer Vision and Pattern Recognition (CVPR). pp. 9924–9935 (2022)

12. Irvin, J., Rajpurkar, P., Ko, M., et al.: Chexpert: A large chest radiograph dataset
with uncertainty labels and expert comparison. In: AAAI conference on artificial
intelligence. vol. 33, pp. 590–597 (2019)

13. Jack Jr, C.R., Bernstein, M.A., Fox, N.C., et al.: The alzheimer’s disease neu-
roimaging initiative (adni): Mri methods. Journal of Magnetic Resonance Imaging
27(4), 685–691 (2008)

14. Levakov, G., Rosenthal, G., Shelef, I., Riklin Raviv, T., Avidan, G.: From a deep
learning model back to the brain—Identifying regional predictors and their relation
to aging. Human Brain Mapping 41(12), 3235–3252 (2020)

15. Liang, J., Hu, D., Feng, J.: Do we really need to access the source data? source hy-
pothesis transfer for unsupervised domain adaptation. In: International conference
on machine learning. pp. 6028–6039. PMLR (2020)

16. Littwin, G., Wolf, L.: Deep meta functionals for shape representation. In: Interna-

tional Conference on Computer Vision (ICCV). pp. 1824–1833 (2019)

17. Liu, W., Wei, D., Chen, Q., et al.: Longitudinal test-retest neuroimaging data from

healthy young adults in southwest china. Scientific data 4(1), 1–9 (2017)

10

D. Serebro et al.

18. Ma, T., Dalca, A.V., Sabuncu, M.R.: Hyper-convolution networks for biomedical
image segmentation. In: Winter Conference on Applications of Computer Vision
(WACV). pp. 1933–1942 (2022)

19. Marcus, D.S., Wang, T.H., Parker, J., et al.: Open access series of imaging studies
(oasis): cross-sectional mri data in young, middle aged, nondemented, and de-
mented older adults. Journal of cognitive neuroscience 19(9), 1498–1507 (2007)
20. Marek, K., Jennings, D., Lasch, S., et al.: The parkinson progression marker initia-
tive (ppmi). Progress in Neurobiology 95(4), 629–635 (2011), biological Markers
for Neurodegenerative Diseases

21. Nguyen, H.Q., Lam, K., Le, L.T., et al.: Vindr-cxr: An open dataset of chest x-rays

with radiologist’s annotations. Scientific Data 9(1), 429 (2022)

22. Nooner, K.B., Colcombe, S.J., Tobe, R.H., et al.: The nki-rockland sample: a model
for accelerating the pace of discovery science in psychiatry. Frontiers in neuroscience
6, 152 (2012)

23. Poldrack, R.A., Congdon, E., Triplett, W., et al.: A phenome-wide examination of

neural and cognitive function. Scientific data 3(1), 1–12 (2016)

24. Rajpurkar, P.: Chexnet: Radiologist-level pneumonia detection on chest x-rays with

deep learning. ArXiv preprint (2017)

25. Shafto, M.A., Tyler, L.K., Dixon, M., et al.: The cambridge centre for ageing and
neuroscience (cam-can) study protocol: a cross-sectional, lifespan, multidisciplinary
examination of healthy cognitive ageing. BMC neurology 14, 1–25 (2014)

26. Shamsian, A., Navon, A., Fetaya, E., Chechik, G.: Personalized federated learning
using hypernetworks. In: International Conference on Machine Learning (ICML).
pp. 9489–9502. PMLR (2021)

27. Sun, B., Saenko, K.: Deep coral: Correlation alignment for deep domain adaptation.
In: Computer vision–ECCV 2016 workshops: Amsterdam, part III 14. pp. 443–450.
Springer (2016)

28. Wang, D., Shelhamer, E., Liu, S., et al.: Tent: Fully test-time adaptation by entropy
minimization. In: International Conference on Learning Representations (2021)
29. Wang, X., Peng, Y., Lu, L., et al.: Chestx-ray8: Hospital-scale chest x-ray database
and benchmarks on weakly-supervised classification and localization of common
thorax diseases. In: Conference on Computer Vision and Pattern Recognition
(CVPR). pp. 2097–2106 (2017)

30. Wang, X., Han, X., Huang, W., Dong, D., Scott, M.R.: Multi-similarity loss with
general pair weighting for deep metric learning. In: Conference on Computer Vision
and Pattern Recognition (CVPR). pp. 5022–5030 (2019)

31. Yang, G., Tang, H., Zhong, Z., et al.: Transformer-based source-free domain adap-

tation. arXiv preprint arXiv:2105.14138 (2021)

32. Zhang, M., Levine, S., Finn, C.: Memo: Test time robustness via adaptation and
augmentation. Advances in Neural Information Processing Systems (NeurIPS) 35,
38629–38642 (2022)

33. Zhao, H., Zhang, S., Wu, G., et al.: Adversarial multiple source domain adaptation.

Advances in Neural Information Processing Systems (NeurIPS) 31 (2018)


