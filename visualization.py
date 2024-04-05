import main
import pandas as pd
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", color_codes=True)

data = main.DataSetCreator().create_vector_list("data")
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','Language'])
print(df["Language"].value_counts())
sns.pairplot(df, hue="Language", size=3)

plt.show()