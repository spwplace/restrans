import { Card, CardContent } from "@/components/ui/card";
import { Image } from "lucide-react";

const base = import.meta.env.BASE_URL;

const figures = [
  {
    src: `${base}fig_architecture.png`,
    caption: "Figure 1: Resonance Transformer Architecture",
    description:
      "The standard transformer (inset, top-right) uses a single embedding stream. The Resonance Transformer splits each token into a semantic stream (blue) and a phase stream (dark blue), blends them via a learnable sigmoid gate, and biases attention with the pairwise resonance matrix R.",
  },
  {
    src: `${base}fig_data_pipeline.png`,
    caption: "Figure 2: Lambda Calculus Proof-Walk Data Pipeline",
    description:
      "Synthetic data generation pipeline: random well-typed lambda terms are generated, normalized reduction walks are performed, and proof states are recorded as token sequences for contrastive pretraining.",
  },
  {
    src: `${base}fig1_scaling_law.png`,
    caption: "Figure 3: Scaling Law — Parameters vs. Perplexity",
    description:
      "Log-log plot of parameter count versus validation perplexity for Standard (circles, solid) and Resonance (squares, dashed) transformers at 1×, 2×, and 4× scales. Both architectures follow approximately linear scaling trends.",
  },
  {
    src: `${base}fig2_compressibility.png`,
    caption: "Figure 4: Compressibility Analysis",
    description:
      "Perplexity degradation under uniform quantization (8-bit, 4-bit, 2-bit) and magnitude pruning (30%, 50%, 70% sparsity). Key result: 2-bit quantization of phase embeddings improves perplexity by 0.24 points while equivalent semantic quantization degrades it by 38.49.",
  },
  {
    src: `${base}fig3_gestalt.png`,
    caption: "Figure 5: Gestalt Stream Geometry",
    description:
      "PCA variance explained curves for semantic and phase embedding streams. The phase stream requires only 29 principal components for 95% variance versus 234 for the semantic stream—an eightfold reduction in effective dimensionality.",
  },
  {
    src: `${base}fig4_perturbation.png`,
    caption: "Figure 6: Perturbation Stability",
    description:
      "Stability ratio (perturbed perplexity / baseline perplexity) under Gaussian noise with σ ∈ [0.001, 0.5]. Phase parameters remain stable (ratio ≈ 1.0) while semantic parameters and full weights degrade monotonically.",
  },
  {
    src: `${base}fig5_training_curves.png`,
    caption: "Figure 7: Training Curves",
    description:
      "Training loss and validation perplexity over epochs for small and medium models. Both architectures exhibit monotonically decreasing training loss and stable validation perplexity without divergence.",
  },
];

export default function Figures() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="flex items-center gap-3 mb-8">
        <Image className="h-6 w-6 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">Figures</h1>
      </div>

      <div className="space-y-10">
        {figures.map((fig) => (
          <Card key={fig.src} className="overflow-hidden">
            <CardContent className="p-0">
              <div className="bg-muted/50 p-6 flex items-center justify-center">
                <img
                  src={fig.src}
                  alt={fig.caption}
                  className="rounded-lg border shadow-sm max-w-full"
                  loading="lazy"
                />
              </div>
              <div className="p-6">
                <h3 className="text-sm font-semibold text-foreground mb-1">
                  {fig.caption}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {fig.description}
                </p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
