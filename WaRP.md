# WaRP Basics 

## 1 Abstract

Weight space Rotation Process (WaRP) is to address the challenges of catastrophic forgetting and overfitting in Class-Incremental Few-Shot Learning (CIFSL).

The core principle of WaRP is to transform the original parameter space into a new, more effective space through a basis transformation. This transformation leverages the low-rank property of activations from previously learned tasks, using SVD to construct a new orthonormal basis.

In this new space, knowledge from previous tasks could be compactly consolidated into a few "important" parameters. To accommodate new classes, WaRP identifies and freezes these key parameters, thereby preserving prior knowledge. The remaining parameters, which correspond to flat directions in the loss landscape, are then fine-tuned on the new few-shot data. This selective update strategy allows the model to effectively learn new information without disrupting established knowledge.

This approach offers a more generalized and flexible framework compared to prior methods that operate within the null space of feature covariances. While those methods often freeze or update parameters in a structured, column-wise manner, WaRP performs this selection on an element-wise basis within the reparameterized space. This allows for a more granular and adaptable method of knowledge preservation and incremental learning.

## 2 Basic Concept of WaRP

### 2.1 Weight Space Rotation

Introduce the general concept of WaRP. A weight matrix $W \in \mathbb{R}^{m \times n}$ of a neural network layer can be interpreted as a linear combination of standard basis matrices. This is formally expressed as:

$$
W=\sum_{i=1}^{m} \sum_{j=1}^{n} w_{i j} e_{i j}
$$

where $\left\{e_{i j}\right\}$ is the set of standard basis matrices, and the learnable parameters $\left\{w_{i j}\right\}_{i j}$ are their corresponding coefficients.

The WaRP framework generalizes this concept by viewing the space of matrices as a Hilbert space, $\mathcal{H}:=\mathbb{R}^{m \times n}$, equipped with the inner product

$\langle A, B\rangle:=\operatorname{trace}\left(A^{\top} B\right)$. Within this space, the weight matrix $W$ can be reparameterized using any complete orthonormal basis $\mathcal{B}=\left\{\mathcal{K}_{i j}\right\}$, which satisfies (i) $\left\langle\mathcal{K}, \mathcal{K}^{\prime}\right\rangle=0$ if $\mathcal{K} \neq \mathcal{K}^{\prime}$ for $\mathcal{K}, \mathcal{K}^{\prime} \in \mathcal{B}$, and (ii) $\|\mathcal{K}\|=\sqrt{\langle\mathcal{K}, \mathcal{K}\rangle}=1$ for all $\mathcal{K} \in \mathcal{B}$, where $\|\cdot\|$ and $\langle\cdot, \cdot\rangle$ are the norm and inner product properly defined in $\mathbb{R}^{m \times n}$, respectively. The matrix is then represented as a linear combination of these new basis matrices:

$$
W=\sum_{i=1}^{m} \sum_{j=1}^{n}\left\langle W, \mathcal{K}_{i j}\right\rangle \mathcal{K}_{i j}
$$

This reparameterization allows for the rotation of weight space, enabling the model's knowledge to be represented along a new set of axes defined by $\mathcal{B}$. The coefficients $\left\langle W, \mathcal{K}_{i j}\right\rangle$ becomes the new parameters of the model in this transformed space.

# 2.2 Constructing New Basis for Weight Space Rotation 

To find an appropriate basis to reparameterize $W \in \mathbb{R}^{m \times n}$, we start by expressing the basis by using a pair of unitary matrices. Given any two unitary matrices $V \in \mathbb{R}^{m \times m}, U \in \mathbb{R}^{n \times n}$, it can be seen that the set $\mathcal{B}=\left\{\mathcal{K}_{i j}:=\mathbf{v}_{i} \mathbf{u}_{j}^{\top} \in\right.$ $\left.\mathbb{R}^{m \times n}|i \in[m], j \in[n]\right\}$ is orthonormal, where $\mathbf{v}_{i}$ and $\mathbf{u}_{j}$ denote the $i$-th and $j$-th column vectors of $V$ and $U$, respectively. Thus from (2), we can rewrite $W$ as,

$$
W=\sum_{i=1}^{m} \sum_{j=1}^{n}\left\langle W, \mathcal{K}_{i j}\right\rangle \mathcal{K}_{i j}=\sum_{i=1}^{m} \sum_{j=1}^{n}\left\langle W, \mathbf{v}_{i} \mathbf{u}_{j}^{\top}\right\rangle \mathbf{v}_{i} \mathbf{u}_{j}^{\top}=\sum_{i=1}^{m} \sum_{j=1}^{n} \widetilde{w}_{i j} \mathbf{v}_{i} \mathbf{u}_{j}^{\top}
$$

Thus, the weight matrix $W$ is reparameterized by the coefficients $\left\{\widetilde{w}_{i j}\right\}_{i, j}$ on the basis $\left\{\mathbf{v}_{i} \mathbf{u}_{j}^{\top}\right\}_{i, j}$, which is formed from two arbitrary unitary matrices $V$ and $U$, with each coefficient $\widetilde{w}_{i j}$ defined as $\left\langle W, \mathbf{v}_{i}, \mathbf{u}_{j}^{\top}\right\rangle$. And the important question is that how could we design appropriate $V$ and $U$ to make most of the elements in the basis to be flat direction.

The primary objective of weight space rotation is to reparameterize the model's coefficients onto a new, more effective basis. The choice of the basis $\mathbf{v}_{\mathbf{i}}$ and $\mathbf{u}_{\mathbf{j}}$ - is crucial as it is task-dependent and directly determines the new parameter space.

## 3 Proposed Algorithm for CIFSL

### 3.1 Problem Setup

In the CIFSL setting, the model is sequentially trained on a series of tasks, where each task introduces a new set of disjoint classes, denoted by $\mathcal{C}_{1}, \ldots, \mathcal{C}_{n}$. The main objective is to learn these new classes effectively while preventing catastrophic forgetting of previously learned informations. A key constraint of this setup is that during the training for a new task $\mathcal{T}_{k}$, the training samples

from all previous tasks $\left\{\mathcal{T}_{i}\right\}_{i=1}^{k-1}$ are inaccessible, reflecting practical limitations such as data privacy and storage constraints. For each $k$-th session, the model is provided with a small training set $\mathcal{T}_{k}=\left\{\left(x_{i}^{k}, y_{i}^{k}\right)\right\} t=1^{N_{k}}$, where $N_{k}$ is the number of samples. The weigth matrix of the $l$-th layer in the neural network is denoted as $W_{l} \in \mathbb{R}^{m_{l} \times n_{l}}$.

In the context of CIFSL, preserving previously acquired knowledge is paramount. To achieve this, the model's weight matrix, $W$, is not updated in its original standard basis. Instead, the weight space is transformed by constructing a new, task-specific basis derived from $\Phi$, a matrix containing the activations from prior tasks. By reparameterizing the weight matrix onto this new basis, the model can isoloate parameters that are critical for past knowledge. This allows for fine-tuning on new classes along directions that are 'flat' with respect to the old tasks, thereby mitigating catastrophic forgetting.

# 3.2 Constructing New Basis for Weight Space Rotation 

For a specific layer $l$, we define a matrix $\Phi_{l}=\left[\phi_{l}\left(x_{1}\right), \phi_{l}\left(x_{2}\right), \ldots, \phi_{l}\left(x_{N_{1}}\right)\right]$ where $\phi_{l}\left(x_{i}\right)$ denotes the activation of the $l$-th layer for the $t$-th sample in the first task $\mathcal{T}_{1}$. The core of WaRP's basis construction lies in performing SVD on the covariance of these activations.

$$
\Phi_{l} \Phi_{l}^{\top}=U_{l} \Sigma_{l} U_{l}^{\top}
$$

Here, the resulting unitary matrix $U_{l}$ is crucial. Its column vectors, $\left\{\mathbf{u}_{j}\right\}$, form an orthonormal basis where each vector's importance is orderd according to the corresponding singular values $\Sigma_{l}$. Adopting this $U_{l}$ as part of the new basis for the weight space means the weight parameters will be reparameterized around the principal directions of the base task's activations.

This design has a significant impact on the forward pass. For any sample $x$ from $\mathcal{T}_{1}$, the layer's output can be expressed as:

$$
W_{l} \phi_{l}(x)=\sum_{i=1}^{m} \sum_{j=1}^{n} \tilde{w}_{i j}^{l} \mathbf{v}_{i} \underbrace{\mathbf{u}_{j}^{\top} \phi_{l}(x)}_{(a) \approx \mathbf{0} \text { for most }(i, j) \text { pairs by low-rankness }}
$$

Due to the low-rank property of activations, any activation $\phi_{l}(x)$ will have negligible components along the directions spanned by the majority of the basis vectors in $U_{l}$ (those corresponding to near-zero singular values). Consequently, the term (a) - the projection of the activation onto these basis vectors - becomes approximately zero for most of pairs. This implies that most directions in the new basis are 'flat' with respect to the base task's loss landscape. Therefore, by identifying and freezing the few parameters corresponding to the meaningful, non-flat directions, we can fine-tune the remaining parameters on new tasks without causing catastrophic forgetting.

# 3.3 Identifying Important Parameters 

Unlike existing null space projection methods, WaRP's distinct feature is the reparameterization of the weight matrix itself, guided by the activations that hold previously learned knowledge. This involves reconstructing the basis of the weight space using these activations and then redefining the weight coefficients with respect to this new basis.

Within this reparameterized space, primary goal of WaRP is to identify 'flat direction' - axes along which modifications do not significantly impact the preserved knowledge. The core mechanism for this is to assess the importance of each new parameters, $\widetilde{w}$, by calculating its gradient with respect to the loss function. This allows the framework to quantify how significantly each individual parameter influences the final loss. Based on these importance scores, WaRP can selectively freeze parameters crucial for past knowledge while fine-tuning the remaining ones to learn new classes.

More in specific, at the end of each $k$-th session, we can induce the score criterion that is well compatible with new space constructed by WaRP using chain rule to identify the trainable parameters, .i.e., flat directions.

$$
d L / d \widetilde{w}_{i j}^{l}=d z_{l}^{\top} \mathbf{v}_{i} \underbrace{\mathbf{u}_{j}^{\top} \phi_{l}(x)}_{(a)}
$$

where $d z_{l}$ denotes the gradient of the loss with respect to the output of the layer $z_{l}=W_{l} \phi_{l}(x)$ for any $x$ in $\mathcal{T}_{k}$. Interestingly, it turns out that the term $(a)$ in equation (6) also appears in (5). In addition, as shown in (6), the trend of the magnitude of $d L / d \widetilde{w}_{i j}^{l}$ follows the tendency of $(a)$. From these observations, it can be seen that the gradient can indirectly capture the influence of finetuning within each direction $\mathbf{v}_{i} \mathbf{u}_{j}^{\top}$ on $\mathcal{T}_{k}$. Motivated by this, we consider the following score criterion:

$$
s_{i j}^{l}:=\text { importance score of } \widetilde{w}_{i j}^{l}=\sum_{b \in \mathcal{D}_{k}^{n_{b}}}\left|d L_{k}(b) / d \widetilde{w}_{i j}^{l}\right|
$$

where $L_{k}(b)$ is the loss computed with batch $b$ in $\mathcal{T}_{k}$, and $\mathcal{D}_{k}^{n_{b}}$ is a set that consists of $N_{k} / n_{b}$ batches with size $n_{b}$ in $\mathcal{T}_{k}$.

This element-wise selection mechanism provides greater flexibility compared to traditional null space projection methods. While those approaches typically constrain updates on a structural, column-wise basis, WaRP allows for a more granular identification of important parameters based on their individual impact on the loss landscape, making it a more generalized framework for continuous learning.

## 4 Comparison against Null Space Projection Method

This approach of constructing basis from activations that encode preserved knowledge seems conceptually analogous to null space projection methods. Both

strategies aim to isolate a protected subspace containing critical past information, thereby defining a complementary, 'safe' subspace where a model can be updated to learn new tasks without catastrophic forgetting.

However, the primary distinction lies in the mechanism and flexibility of this preservation. Gradient projection methods enforce a structural constraint, projection the entire update vector into a null space that is orthogonal to the protected feature space. In contrast, WaRP offers a more granular, element-wise control by identifying and freezing individual parameters within its reparameterized space, which provides a more flexible framework for knowledge retention.

# 4.1 Mathematical Proof 

In this section, we first formalize how the classical null-space projection method guarantees preservation of past knowledge, and then show that WaRP generalizes the same preservation principle to a more fine-grained (element-wise) level.

Setup. Let the activation matrix of the $l$-th layer collected from the base session be

$$
\Phi_{l}=\left[\phi_{l}\left(x_{1}\right), \ldots, \phi_{l}\left(x_{N_{1}}\right)\right] \in \mathbb{R}^{n_{l} \times N_{1}}
$$

For the covariance matrix $C_{l}:=\Phi_{l} \Phi_{l}^{\top}$, take its SVD

$$
C_{l}=U_{l} \Sigma_{l} U_{l}^{\top}, \quad U_{l}=\left[\mathbf{u}_{1}, \ldots, \mathbf{u}_{k} \mid \mathbf{u}_{k+1}, \ldots, \mathbf{u}_{n_{l}}\right]=[Q \mid P]
$$

where $Q \in \mathbb{R}^{n_{l} \times k}$ collects columns associated with large singular values (important directions), and $P \in \mathbb{R}^{n_{l} \times\left(n_{l}-k\right)}$ spans the null space corresponding to small or near-zero singular values. We have $Q Q^{\top}+P P^{\top}=I$ and $Q^{\top} P=\mathbf{0}$. Denote the weight matrix $W_{l} \in \mathbb{R}^{m_{l} \times n_{l}}$, the layer output $z_{l}=W_{l} \phi_{l}(x)$, and the gradient $G_{l}:=\frac{\partial L}{\partial W_{l}} \in \mathbb{R}^{m_{l} \times n_{l}}$.

### 4.1.1 Null-space projection

Classical methods (e.g., OGD/GPM-style updates) preserve the past subspace $\operatorname{span}(Q)$ by projecting the gradient into the null space spanned by $P$ :

$$
\Delta W_{l}=-\eta G_{l} \Pi_{P}, \quad \text { where } \quad \Pi_{P}:=P P^{\top}=I-Q Q^{\top}
$$

For any past sample $x$ with $\phi_{l}(x) \in \operatorname{span}(Q)$,

$$
\Delta z_{l}=\Delta W_{l} \phi_{l}(x)=-\eta G_{l} \Pi_{P} \phi_{l}(x)=-\eta G_{l} \underbrace{\left(I-Q Q^{\top}\right) \phi_{l}(x)}_{=0}=0
$$

so past representations are exactly unchanged (zero interference). Moreover,

$$
\Delta W_{l} U_{l}=\Delta W_{l}[Q \mid P]=\left[\Delta W_{l} Q \mid \Delta W_{l} P\right]=[0 \mid-\eta G_{l} P]
$$

Thus, in the $U_{l}$ coordinates, all columns with index $j \leq k$ (the $Q$-block) are simultaneously zeroed. In other words, the classical approach imposes a columnwise constraint that freezes an entire set of columns selected by a singular-value threshold.

# 4.1.2 WaRP: element-wise preservation in rotated coordinates 

As described in Sections 3.2 and 3.3, WaRP reparameterizes $W_{l}$ on the basis $\left\{\mathbf{v}_{i} \mathbf{u}_{j}^{\top}\right\}_{i, j}$ generated by two unitary matrices $V_{l} \in \mathbb{R}^{m_{l} \times m_{l}}$ and $U_{l} \in \mathbb{R}^{n_{l} \times n_{l}}$ :

$$
W_{l}=\sum_{i=1}^{m_{l}} \sum_{j=1}^{n_{l}} \widetilde{w}_{i j}^{l} \mathbf{v}_{i} \mathbf{u}_{j}^{\top}, \quad \widetilde{G}_{l}:=V_{l}^{\top} G_{l} U_{l}
$$

Here, $\widetilde{G}_{l}$ is the gradient expressed in the rotated coordinates. WaRP updates apply an element-wise mask $M \in\{0,1\}^{m_{l} \times n_{l}}$ derived from the importance scores in (7):

$$
\Delta \widetilde{W}_{l}=-\eta\left(M \circ \widetilde{G}_{l}\right), \quad \Delta W_{l}=V_{l} \Delta \widetilde{W}_{l} U_{l}^{\top}=-\eta V_{l}\left(M \circ \widetilde{G}_{l}\right) U_{l}^{\top}
$$

where o denotes the Hadamard (element-wise) product. Because important directions can be frozen at the individual element level $(i, j)$ (by setting $M_{i j}=0$ ), WaRP provides much finer, element-wise control than the column-wise constraint of the classical method.

Recovering the classical method as a special case. Let $c \in\{0,1\}^{n_{l}}$ be a column mask and set $M=\mathbf{1} c^{\top}$ (the same column mask for all rows). For any $A \in \mathbb{R}^{m_{l} \times n_{l}}$,

$$
A \circ\left(\mathbf{1} c^{\top}\right)=A \operatorname{diag}(c)
$$

Applying (11) to (10) yields

$$
\Delta W_{l}=-\eta V_{l} \widetilde{G}_{l} \operatorname{diag}(c) U_{l}^{\top}=-\eta\left(V_{l} \widetilde{G}_{l} U_{l}^{\top}\right) \underbrace{U_{l} \operatorname{diag}(c) U_{c}^{\top}}_{=: \Pi_{c}}=-\eta G_{l} \Pi_{c}
$$

With $c=(\underbrace{0, \ldots, 0}_{k}, \underbrace{1, \ldots, 1}_{n_{l}-k})$, we have $\Pi_{c}=P P^{\top}=\Pi_{P}$, which exactly matches
(9). Hence, WaRP reproduces the classical null-space update when a column-wise mask is chosen, while in general it allows arbitrary element-wise masks for finer control.

### 4.2 Compare with Other Methods

Recent advancements in learning algorithms have explored leveraging the algebraic properties of neural network activations to mitigate catastrophic forgetting and preserve existing knowledge. A shared, fundamental principle among several state-of-the-art methods is the strategic use of the null space of activation vectors. Although their specific algorithms differ, approaches such as Gradient Projection Memroy (GPM), WaRP, and AlphaEdit all converge on the idea of constraining model updates to this null space, ensuring that modifications for new knowledge do not interfere with previously learned representations.

Gradient Projection Memory GPM operates on the principle that the gradient of the weights $\nabla_{W} L$ at any given layer lies within the span of its input activation $x$. To prevent new learning from interfering with past tasks, GPM directly constrains the gradient updates.

- Mechanism: After learning a task, GPM uses SVD on the matrix of activation to identify a low-dimensional basis for the most significant representations. This basis spans define as the Core Gradient Space (CGS), which deemed critical for retaining past knowledge.
- Execution: When learning a subsequent task, the newly computed gradients are projected into the orthogonal complement of the CGS -effectively, the null space- before the weight update is applied. By ensuring the gradient update is orthogonal to the subspace of important past activations, GPM minimizes interference and mitigate catastrophic forgetting.

AlphaEdit AlphaEdit applies a similar null-space constraint within the context of model editing, which frames knowledge as key-value associations in the feed-forward network (FFN) layers. In this paradigm, the keys correspond to activations.

- Mechanism: The primary objective is to udpate the model's parameters by adding a perturbation $\Delta$ such that new knowledge is incorporated while preserved knowledge remains unchanged. For the output corresponding to preserved knowledge keys $\left(K_{0}\right)$ to remain invariant, the perturbation must satisfy the condition ${ }_{0}=0$, meaning $\Delta$ must lie in the left null space of the activation matrix $K_{0}$.
- Execution: AlphaEdit pre-computes a projection matrix that maps any calculated perturbation $\Delta$ into the null space of the preserved keys $K_{0}$. This allows the optimization to focus solely on minimizing the error for the new knowledge, as the projection mathematically guarantees that the update will not affect the outputs for any knowledge the model is intended to preserve.

WaRP (Warping the Space) Unlike GPM and AlphaEdit, which project the update vector, WaRP achieves the same goal by reparameterizing the weight space itself based on the properties of the activations.

- Mechanism: WaRP leverages the low-rank property of neural network activations, where most information is concentrated in a few principal directions. It uses SVD on the activation covariance matrix to construct a new orthonormal basis for the weight space.
- Execution: In this new warped space, the reparameterized weights are separated into two groups. The important parameters which correspond to the principal components of the activations, are frozen to preserve existing knowledge. The remaining parameters, which correspond to the

null space of the activations (which could be termed as 'flat direction' in the loss landscape), are fine-tuned to learn new classes. Updating along these flat directions ensures that the model can acquire new information without performance degradation on previous tasks.

Summary These three methods, while algorithmically distinct, are founded on the same core insight, to preserve learned knowledge associated with a set of activations, any subsequent weight modification must be confined to the null space of those activations. GPM and AlphaEdit achieve this through direct projection of the update vector (gradient and parameter perturbation, respectively). In contrast, WaRP achieves it through weight space reparameterization and selective updating, which implicitly restricts changes to the parameters that operate within the activation null space. This convergence of strategies underscores the foundational importance of the activation null space in developing robust and efficient methods for continual learning and model editing.

