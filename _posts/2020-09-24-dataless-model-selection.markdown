---
layout: post
mathjax: true
comments: true
title:  "IN PROGRESS Overview of Dataless Model Selection with the Deep Frame Potential"
date:   2020-09-24 20:00:00 -0400
categories: jekyll update
---
Today I'll be giving my own overview and interpretation of the following publication: 

 - [Murdock, Calvin, and Simon Lucey. "Dataless Model Selection with the Deep Frame Potential." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.](https://openaccess.thecvf.com/content_CVPR_2020/html/Murdock_Dataless_Model_Selection_With_the_Deep_Frame_Potential_CVPR_2020_paper.html)

*Disclaimer: This post is a representation of my interpretation and opinions of the aforementioned work and should not be taken, in any way, to represent the opinions or views of the original authors.* 

### Overview

This paper describes a process of quantifying the performance of a network without training or validation data. To this end
the authors introduce the "deep frame potential" measure. By utilizing previously researched connections between deep
learning and sparse representations, they measure the coherence of an architecture and provide theoretical arguments along
with quantitative evidence demonstrating the efficacy of their approach.

#### Mutual-coherence

Mutual-coherence of a dictionary $$D \in \mathbb{R}^{d \times k}$$ is defined as follows

Let $$\hat{d}_i \triangleq \frac{1}{d_i^T d_i} d_i$$ where $$d_i$$ is the $$i^{th}$$ column of $$D$$. Then mutual-coherence of
$$D$$ is defined as

$$
M(D) \triangleq \max_{1 \leq i \neq j \leq k} \left| \hat{d}_i^T \hat{d}_j \right|
$$

*Personal observations about mutual-coherence:* Since I'm not very familiar with dictionary learning, I'll put my interpretation
of mutual-coherence here:

- If $$M = 0$$ then the columns of $$D$$ form an orthogonal basis. This is ideal for dictionary learning since it
ensures the dictionary does not contain redundant information and also ensures that any representation is unique. By
this I mean that for some representation $$r$$, there does not exist any $$\hat{r} \neq r$$ for which $$Dr = D\hat{r}$$.
In general, we often have dictionaries that are over-complete, i.e. $$k > d$$. In such cases we cannot achieve $$M = 0$$
and instead may try to define a dictionary $$D$$ for which $$M$$ is minimal.

- This can be interpreted as a way of ensuring that the dictionary contains as little redundancy as possible. Another way
to think about this is that the atoms (i.e. columns of $$D$$) are as evenly distributed as possible on the surface of the unit hypersphere in $$\mathbb{R}^d$$.
According to the authors of the paper, if atoms in a dictionary are close to eachother then there may be
"instabilities" in the representation. By this I assume they mean it's more likely for two sparse representations of the
same or similar data to be far apart (according to some distance metric). 


#### Context

The authors have shown their previous work:

- [Murdock, Calvin, MingFang Chang, and Simon Lucey. "Deep component analysis via alternating direction neural networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.](https://openaccess.thecvf.com/content_ECCV_2018/html/Calvin_Murdock_Deep_Component_Analysis_ECCV_2018_paper.html)

that, given proper initilaization of variables, evaluating a feed-forward network

$$
\mathbf{f}^{DNN}(\mathbf{x}) = \phi_l(\mathbf{B}_l^T \cdots \phi_2(\mathbf{B}_2^T(\phi_1(\mathbf{B}_1^T\mathbf{x} - \mathbf{b}_1) - \mathbf{b}_2) \cdots - \mathbf{b}_l)
$$

can be interpreted as a single step in the Alternating Direction Method of Multipliers (ADMM) applied to the following optimization problem:

$$
\mathbf{f}^*(\mathbf{x}) = \underset{\left\{\mathbf{w_j}\right\}}{\text{arg min}}{\sum_{j=1}^{l} \frac{1}{2} || \mathbf{w}_{j-1} - \mathbf{B}_j \mathbf{w}_j||^2_2 + \Phi_j(\mathbf{w}_j)}~~~\text{s.t.}~\mathbf{w}_0 = \mathbf{x}
$$

Here, $$\mathbf{w}_*$$ are the activations of each layer and $$\Phi_*$$ is a penalty function. The authors show that
a particular form of $$\Phi_*$$ and particular initialization of variables makes it so a single step of ADMM is identical to
$$\mathbf{f}^{DNN}$$. Further, they extend this idea, taking multiple steps and imposing additional constraints that would difficult
or potentially impossible in traditional DCNNs. For example, they investigate imposing a prior on depth estimation where a subset of depths
are already known. The impressive results of this are shown in Figure 3 of their paper.