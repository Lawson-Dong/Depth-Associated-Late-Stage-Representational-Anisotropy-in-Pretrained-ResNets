# Depth-Associated Late-Stage Representational Anisotropy in Pretrained ResNets

## Research background

Take the same set of images and pass them through a frozen vision model. At each residual block, we can read out one feature vector per image. Put those vectors side by side and we have a point cloud. Repeat this at the next block and we can ask how the cloud has changed, even though the input images are exactly the same.

One question is how the variation in that cloud is distributed. Do many directions carry comparable amounts of variance, or do a few directions dominate? Here we use that question to study the late stages of pretrained ResNets. We track the same image batches through five models and compare the last block of Layer 3 with the last block of Layer 4. The term *depth-associated* describes the scope of the comparison across these checkpoints.

## What does participation ratio measure?

Let $Z\in\mathbb{R}^{N\times C}$ contain the centered feature vectors at one block, one image per row. The sample covariance is $\Sigma = Z^\top Z/(N-1)$. Write its eigenvalues as $\lambda_1,\ldots,\lambda_C$. For a cloud with nonzero variance, the **participation ratio (PR)** is

$$
\mathrm{PR}(Z)
= \frac{\left(\sum_{k=1}^{C}\lambda_k\right)^2}{\sum_{k=1}^{C}\lambda_k^2}
= \frac{\bigl(\operatorname{tr}\Sigma\bigr)^2}{\lVert\Sigma\rVert_F^2}
= \frac{1}{\sum_{k=1}^{C}p_k^2},
\qquad p_k=\frac{\lambda_k}{\sum_j\lambda_j}.
$$

Think of $p_k$ as the share of variance carried by principal direction $k$. If four directions each carry a quarter of the variance, PR is 4. If nearly all the variance sits in one direction, PR is close to 1. More generally, $1\leq\mathrm{PR}\leq\min(C,N-1)$, and the upper bound can be reached when the available nonzero eigenvalues are equal.

In this experiment, a lower PR means that the covariance spectrum of the measured representation is more concentrated. Since PR depends on the *shares* of variance, a uniform rescaling of the centered cloud leaves it unchanged. PR measures effective dimensionality at the level of the covariance spectrum. Estimating the intrinsic dimension of a nonlinear manifold or measuring classification quality would require a different readout.

## From block activations to Overall PR

1. **Read the complete block output.** Use the preprocessing associated with the ImageNet-1K V1 weights. Run the model in evaluation mode and capture the output tensor of each complete residual block.
2. **Make one vector per image.** Apply global average pooling over the spatial axes of the block tensor $A\in\mathbb{R}^{N\times C\times H\times W}$. This gives $v_{ic}=\frac{1}{HW}\sum_{h,w}A_{ichw}$. Normalize each image vector by its L2 norm, $x_i=v_i/\lVert v_i\rVert_2$.
3. **Center the two classes together.** For each class pair and block, stack the 200 normalized vectors and subtract their common mean, $z_i=x_i-\frac{1}{N}\sum_j x_j$. This keeps variation within each class and the contribution from separation between class means in the same covariance. That joint calculation is what *Overall PR* means here.
4. **Compute PR, then average across pairs.** Use the smaller of $ZZ^\top$ and $Z^\top Z$. These matrices have the same nonzero eigenvalues, and the covariance scaling cancels from the ratio, so the trace and Frobenius norm give the same PR without an eigendecomposition. Compute one PR per pair per block, then average the 100 values.

## Experiments

[Experiment 1 — Overall PR across 100 class pairs](experiment_1/README.md) contains the Colab notebook, the sampling protocol, and both result figures.

## References

- He et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Ansuini et al. (2019). [Intrinsic Dimension of Data Representations in Deep Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/cfcce0621b49c983991ead4c3d4d3b6b-Abstract.html).
- Recanatesi et al. (2019). [Dimensionality Compression and Expansion in Deep Neural Networks](https://arxiv.org/abs/1906.00443).
