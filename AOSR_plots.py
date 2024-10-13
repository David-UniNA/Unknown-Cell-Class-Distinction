import matplotlib.pyplot as plt                                             # Imports the matplotlib library for creating static, animated, and interactive visualizations in Python.


def plot_AOSR_results(Test=None, Pred=None, Encoder=None, Encoder_Pred=None, T=None, W=None, plot_type='closed_set'):
    if plot_type == 'closed_set':
# Closed-set Learning Decision Boundary
        plt.scatter(Test[Pred==0, 0], Test[Pred==0, 1])
        plt.scatter(Test[Pred==1, 0], Test[Pred==1, 1])
        plt.scatter(Test[Pred==2, 0], Test[Pred==2, 1])
        plt.scatter(Test[Pred==3, 0], Test[Pred==3, 1])
        plt.yticks(fontsize=13.5)
        plt.xticks(fontsize=13.5)
        plt.title('Closed-set Learning Decision Boundary', fontsize=15)
        plt.legend(['Class1', 'Class2', 'Class3', 'Class4'], ncol=4, fontsize=12)
        plt.savefig("Closeset_pred.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()
# Training samples
    elif plot_type == 'open_set':
        plt.scatter(Encoder[Encoder_Pred==0,0], Encoder[Encoder_Pred==0,1])
        plt.scatter(Encoder[Encoder_Pred==1,0], Encoder[Encoder_Pred==1,1])
        plt.scatter(Encoder[Encoder_Pred==2,0], Encoder[Encoder_Pred==2,1])
        plt.scatter(Encoder[Encoder_Pred==3,0], Encoder[Encoder_Pred==3,1])
        plt.yticks(fontsize=13.5)
        plt.xticks(fontsize=13.5)
        plt.title('Training Samples', fontsize=15)
        plt.legend(['Class1', 'Class2', 'Class3', 'Class4'], ncol=4, fontsize=12)
        plt.savefig("train_data.pdf", bbox_inches = 'tight',pad_inches = 0)
        plt.show()

    elif plot_type == 'uniform_classification':
# Uniform Data with Classification Results
        plt.scatter(Test[Pred==0, 0], Test[Pred==0, 1])
        plt.scatter(Test[Pred==1, 0], Test[Pred==1, 1])
        plt.scatter(Test[Pred==2, 0], Test[Pred==2, 1])
        plt.scatter(Test[Pred==3, 0], Test[Pred==3, 1])
        plt.scatter(Encoder[Encoder_Pred==0, 0], Encoder[Encoder_Pred==0, 1])
        plt.scatter(Encoder[Encoder_Pred==1, 0], Encoder[Encoder_Pred==1, 1])
        plt.scatter(Encoder[Encoder_Pred==2, 0], Encoder[Encoder_Pred==2, 1])
        plt.scatter(Encoder[Encoder_Pred==3, 0], Encoder[Encoder_Pred==3, 1])
        plt.yticks(fontsize=13.5)
        plt.xticks(fontsize=13.5)
        plt.title('Uniform Data with Classification Results', fontsize=15)
        plt.legend(['Class1', 'Class2', 'Class3', 'Class4', 'Train Class1', 'Train Class2', 'Train Class3', 'Train Class4'], ncol=8, fontsize=12)
        plt.savefig("Closeset_pred_with_train.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()

    elif plot_type == 'openset_enrichment':
# Openset Sample Enrichment Result - IF
        plt.scatter(T[W!=0, 0], T[W!=0, 1])
        plt.scatter(T[W==0, 0], T[W==0, 1])
        plt.yticks(fontsize=13.5)
        plt.xticks(fontsize=13.5)
        plt.title('Openset Sample Enrichment Result - IF', fontsize=15)
        plt.legend(['Unknown Samples', 'Known Samples'], ncol=3, fontsize=12)
        plt.savefig("AOSR_Enrich_if.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()
# Openset Sample Classification
    elif plot_type == 'classification':
        plt.scatter(Test[Pred==0,0], Test[Pred==0,1])
        plt.scatter(Test[Pred==1,0], Test[Pred==1,1])
        plt.scatter(Test[Pred==2,0], Test[Pred==2,1])
        plt.scatter(Test[Pred==3,0], Test[Pred==3,1])
        plt.scatter(Test[Pred==4,0], Test[Pred==4,1], c='grey', marker='x')
        plt.yticks(fontsize=13.5)
        plt.xticks(fontsize=13.5)
        plt.title('Classification Results Using Open-set Learning', fontsize=15)
        plt.legend(['Class1', 'Class2', 'Class4', 'Class4', 'Unknowns'], ncol=5, fontsize=12)
        plt.savefig("openset_pred.pdf", bbox_inches = 'tight',pad_inches = 0)
        plt.show()
