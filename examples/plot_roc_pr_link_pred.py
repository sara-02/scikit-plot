"""
An example showing the plot_roc_curve method
used by a scikit-learn classifier
"""
from __future__ import absolute_import
from scikitplot.metrics import plot_roc, plot_precision_recall
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data

X, y = load_data(return_X_y=True)
nb = GaussianNB()
nb.fit(X, y)
probas = nb.predict_proba(X)
plot_roc(y_true=y,
         y_probas=probas,
         plot_micro=False,
         classes_to_plot=[1, 2],
         plot_macro=False,
         text_fontsize=30,
         title_fontsize=30,
         figsize=(30, 24),
         title='(a)')
# plt.savefig("roc_link_pred.svg", dpi=850, bbox_inches="tight")
plt.savefig("roc_link_pred.png", dpi=850, bbox_inches="tight")

plot_precision_recall(y_true=y,
                      y_probas=probas,
                      plot_micro=False,
                      classes_to_plot=[1, 2],
                      text_fontsize=30,
                      title_fontsize=30,
                      figsize=(30, 24),
                      title='(b)')
# plt.xlabel("Recall", fontsize=30)
# plt.ylabel("Precision", fontsize=30)
plt.legend(loc='lower right', fontsize=30)
# plt.savefig("pr_link_pred.svg", dpi=850, bbox_inches="tight")
plt.savefig("pr_link_pred.png", dpi=850, bbox_inches="tight")