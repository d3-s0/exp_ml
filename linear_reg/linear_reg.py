import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import json

def load_and_prepare_data(filepath: str, features: list, target: str, column_filters: dict = None, sep: str = ','):
    df = pd.read_csv(filepath, sep=sep)
    
    # Drop rows where critical variables are completely missing
    df = df.dropna(subset=features + [target])
    if column_filters:
        for column, bounds in column_filters.items():
            if column in df.columns:
                min_b, max_b = bounds
                df = df[(df[column] >= min_b) & (df[column] <= max_b)]
    
    # Clean up standard missing records
    df = df.dropna(subset=features + [target])

    return df[features], df[target]




def evaluate_model_math(y_true: pd.Series, y_pred: np.ndarray):
    corr_coef, p_val = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Pearson Correlation (r) : {corr_coef:.4f}")
    print(f"R-squared Score (R²)    : {r2:.4f} ({r2*100:.1f}% of variance explained)")
    print(f"p-value     : {p_val:.4e}")
    print(f"Avg Prediction Error: {mae:.2f} mmHg")
    
    if p_val < 0.05 and corr_coef > 0.3:
        print("Conclusion: The model is PROVEN to be significantly correlated.")
    else:
        print("Conclusion: Weak or no linear correlation detected.\n")
        
    return {"r": corr_coef, "r2": r2, "mae": mae}


def plot_scatter_with_marginals(y_true: pd.Series, y_pred: np.ndarray, target_label: str, output_filename: str = 'linear_reg.png'):
    sns.set_theme(style="ticks")
    combined_data = np.concatenate([y_true, y_pred])
    min_val = np.percentile(combined_data, 1)   
    max_val = np.percentile(combined_data, 99)  
    grid = sns.JointGrid(height=7, space=0.15)
    grid.ax_joint.scatter(
        y_true, y_pred, 
        alpha=0.12, color="teal", s=12, edgecolor="none"
    )
    sns.regplot(
        x=y_true, y=y_pred, ax=grid.ax_joint, scatter=False, 
        color="darkcyan", line_kws={"linewidth": 3}, 
        label="Model Trend Line"
    )
    grid.ax_joint.plot(
        [min_val, max_val], [min_val, max_val], 
        color="crimson", linestyle="--", linewidth=2.5, 
        label="Perfect Alignment Line"
    )
    sns.histplot(x=y_true, ax=grid.ax_marg_x, bins=35, color="steelblue", fill=True, kde=True)
    sns.histplot(y=y_pred, ax=grid.ax_marg_y, bins=35, color="steelblue", fill=True, kde=True)
    grid.ax_joint.set_xlim(min_val, max_val)
    grid.ax_joint.set_ylim(min_val, max_val)
    grid.set_axis_labels(f"Actual {target_label}", f"Predicted {target_label}", fontsize=11)
    grid.ax_joint.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def print_formula(model, feature_names, target_name: str = "Target"):
    intercept = model.intercept_
    coefficients = model.coef_
    formula_terms = [f"{intercept:.4f}"]
    for name, coef in zip(feature_names, coefficients):
        sign = "+" if coef >= 0 else "-"
        formula_terms.append(f"{sign} ({abs(coef):.4f} * {name})")
    
    full_formula = f"Predicted_{target_name} = " + " ".join(formula_terms)
    print(full_formula)
    
    return full_formula



if __name__ == "__main__":
    with open('spec.json', 'r') as file:
        RUN_CONFIG = json.load(file)

    X, y = load_and_prepare_data(
            filepath=RUN_CONFIG["path"], 
            features=RUN_CONFIG["features"], 
            target=RUN_CONFIG["target"],
            column_filters=RUN_CONFIG.get("filters")
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_model_math(y_test, predictions)
    plot_scatter_with_marginals(y_test, predictions, RUN_CONFIG['label'])

    print_formula(
            model=model, 
            feature_names=RUN_CONFIG["features"], 
            target_name=RUN_CONFIG["target"]
        )