import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(y_train, kde=True)
plt.title('distribution of Target')
plt.show()