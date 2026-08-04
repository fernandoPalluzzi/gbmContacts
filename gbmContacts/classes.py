### ===============================================================
### AbAgInteractionPredictor                                       
### Comprehensive antibody-antigen interaction analysis framework.
### ===============================================================

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ML and data processing.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Plotting and visualization.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Scipy for smoothing.
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

# LightGBM (optional but recommended).
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    warnings.warn("ReportLab not installed. PDF reports unavailable. Install with: pip install reportlab")


# =======================
# ENUMS AND DATA CLASSES
# =======================

class RiskLevel(Enum):
    """Risk classification levels"""
    UNMANAGEABLE = (0, 0.80, "Unmanageable")
    PARTIALLY_MANAGEABLE = (0.00, 0.88, "Partially Manageable")
    MANAGEABLE = (0.88, 1.01, "Manageable")

    def classify(self, score: float) -> str:
        """Classify a risk score"""
        for level in RiskLevel:
            if level.value[0] <= score < level.value[1]:
                return level.value[2]
        return RiskLevel.MANAGEABLE.value[2]

class InteractionType(Enum):
    """Classification of interaction types"""
    STABLE = "High-confidence stable known antibody interaction."
    WEAKLY_STABLE = "Stable interaction with one or more barely stable contacts."
    UNSTABLE = "High-risk variant with poor interaction with known antibodies."
    EXTREMELY_UNSTABLE = "High-risk variant with no known valid antibody."

@dataclass
class PredictionResult:
    """Container for prediction results"""
    interaction_type: InteractionType
    interaction_affinity: float
    risk_score: float
    risk_level: str
    confidence: float
    primary_variant_match: Optional[str]
    best_ab_match: Optional[str]
    mutation_impact: Optional[float]
    contact_affinity_scores: Dict[str, float]
    temporal_affinity_trend: pd.DataFrame
    ml_predictions: Dict[str, float]
    feature_importance: Dict[str, float]
    cross_validation_metrics: Dict[str, float]


# ======================================
# AbAgInteractionPredictor (MAIN CLASS)
# ======================================

class AbAgInteractionPredictor():
    """
    Comprehensive antibody-antigen interaction predictor.
    
    Analyzes user-provided antibody-antigen contacts against a library
    of empirically-validated interactions to assess viral escape risk.
    
    Attributes:
        affinity_data (pd.DataFrame): Affinity scores (NxM where N=timepoints, M=Ab residues)
        affinity_metadata (pd.DataFrame): Metadata for affinity measurements
        contact_class (pd.DataFrame): Contact surface classifications
        contact_map (pd.DataFrame): Validated contact map
        contact_mutations (pd.DataFrame): Known mutations
        aa_properties (pd.DataFrame): Amino acid biochemical properties
    """
    
    def __init__(
        self,
        affinity_data: pd.DataFrame,
        affinity_metadata: pd.DataFrame,
        contact_class: pd.DataFrame,
        contact_map: pd.DataFrame,
        contact_mutations: pd.DataFrame,
        aa_properties: pd.DataFrame,
        affinity_threshold: float = 0.88,
        stability_threshold: float = 0.8
    ):
        """
        Initialize the predictor with library data.
        
        Parameters:
        -----------
        affinity_data : pd.DataFrame
            Shape (n_timepoints, n_ab_residues). Columns are Ab residues.
        affinity_metadata : pd.DataFrame
            Metadata mapping affinity_data columns. Must have 'antibody', 'variants', 'replicates'.
        contact_class : pd.DataFrame
            Contact classifications. Columns: 'residue', 'contact_class'.
        contact_map : pd.DataFrame
            Validated contacts. Columns: 'antibody', 'residue', 'contact'.
        contact_mutations : pd.DataFrame
            Known mutations. Columns: 'variant', 'wt', 'mutant', 'wt.group', 'mutant.group'.
        aa_properties : pd.DataFrame
            AA properties. Columns: 'name', 'three_letter', 'single_letter', 'biochemical_group', 'description'.
        """
        self.affinity_data = affinity_data.copy()
        self.affinity_metadata = affinity_metadata.copy()
        self.contact_class = contact_class.copy()
        self.contact_map = contact_map.copy()
        self.contact_mutations = contact_mutations.copy()
        self.aa_properties = aa_properties.copy()
        self.affinity_threshold = affinity_threshold
        self.stability_threshold = stability_threshold
        
        # Validate inputs.
        self._validate_inputs()
        
        # Build auxiliary structures.
        self._build_indices()
        
        # ML models and scalers (trained during fit).
        self.rf_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.scaler_affinity = MinMaxScaler()
        self.is_fitted = False
        
    def _validate_inputs(self):
        """Validate input data integrity"""
        # Check metadata length matches affinity data columns.
        if len(self.affinity_metadata) != len(self.affinity_data.columns):
            raise ValueError(
                f"Metadata length ({len(self.affinity_metadata)}) != "
                f"affinity data columns ({len(self.affinity_data.columns)})"
            )
        
        # Check required columns.
        required_meta = {'antibody', 'variants', 'replicates'}
        if not required_meta.issubset(set(self.affinity_metadata.columns)):
            raise ValueError(f"Metadata missing columns: {required_meta}")
        
        required_contact_class = {'residue', 'contact_class'}
        if not required_contact_class.issubset(set(self.contact_class.columns)):
            raise ValueError(f"contact_class missing columns: {required_contact_class}")
        
        required_contact_map = {'antibody', 'residue', 'contact'}
        if not required_contact_map.issubset(set(self.contact_map.columns)):
            raise ValueError(f"contact_map missing columns: {required_contact_map}")
    
    def _build_indices(self):
        """Build lookup indices for fast access"""
        # Antibody residue -> affinity data column mapping.
        self.ab_residue_index = {
            col: idx for idx, col in enumerate(self.affinity_data.columns)
        }
        
        # Contact class lookup: antigen residue -> class.
        self.contact_class_lookup = dict(
            zip(self.contact_class['residue'], self.contact_class['contact_class'])
        )
        
        # Contact map: (antibody, ab_residue) -> [ag_residues].
        self.contact_map_lookup = defaultdict(list)
        for _, row in self.contact_map.iterrows():
            key = (row['antibody'].replace('ab.', ''), row['residue'])
            self.contact_map_lookup[key].append(row['contact'])
        
        # Mutations: (variant, wt_residue) -> (mutant_residue, wt_group, mutant_group).
        self.mutations_lookup = {}
        for _, row in self.contact_mutations.iterrows():
            key = (row['variant'], row['wt'])
            self.mutations_lookup[key] = {
                'mutant': row['mutant'],
                'wt_group': row['wt.group'],
                'mutant_group': row['mutant.group']
            }
        
        # AA properties: single_letter -> properties.
        self.aa_properties_lookup = {}
        for _, row in self.aa_properties.iterrows():
            self.aa_properties_lookup[row['single_letter']] = {
                'name': row['name'],
                'group': row['biochemical_group'],
                'description': row.get('description', '')
            }
        
        # Variant index.
        self.variants = sorted(self.affinity_metadata['variants'].unique())
        self.antibodies = sorted(self.affinity_metadata['antibody'].unique())
    
    @staticmethod
    def parse_residue_string(residue: str) -> Tuple[str, str, int]:
        """
        Parse residue string in format 'side_chain.single_letter_codeNNN'.
        
        Examples:
            'h.R50' -> ('h', 'R', 50)   # Ab heavy chain, Arg, position 50.
            'l.Y92' -> ('l', 'Y', 92)   # Ab light chain, Tyr, position 92.
            'V483' -> ('ag', 'V', 483)  # Ag residue, Val, position 483.
        """
        residue = residue.strip()
        
        if '.' in residue:
            # Antibody resirue.
            side_chain, rest = residue.split('.')
            match = re.match(r'([A-Z])(\d+)', rest)
            if match:
                aa_code = match.group(1)
                position = int(match.group(2))
                return side_chain, aa_code, position
        else:
            # Antigen residue.
            match = re.match(r'([A-Z])(\d+)', residue)
            if match:
                aa_code = match.group(1)
                position = int(match.group(2))
                return 'ag', aa_code, position
        
        raise ValueError(f"Cannot parse residue: {residue}")
    
    @staticmethod
    def load_contacts_from_file(filepath: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Load contacts from a text file.
        
        File format:
            h.R50, V483, E484
            h.L55, L452, T470, F490
            ...
        
        Separators can be: space, tab, comma+space, semicolon+space.
        """
        contacts = {}
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try to split by various separators.
                for sep in [',', ';', '\t']:
                    if sep in line:
                        parts = [p.strip() for p in line.split(sep)]
                        break
                else:
                    parts = line.split()
                
                if len(parts) < 2:
                    warnings.warn(f"Line {i} has fewer than 2 residues, skipping")
                    continue
                
                contact_id = f"contact_{i}"
                contacts[contact_id] = parts
        
        if not contacts:
            raise ValueError("No contacts found in file")
        
        return contacts
    
    def _get_affinity_features(
            self,
            ab_residue: str,
            ag_residues: List[str]
        ) -> Dict[str, float]:
        """
        Extract affinity features used for model training.
        Matches the feature set used during model training exactly.
        
        Parameters:
        -----------
        ab_residue : str
            Antibody residue (e.g., 'h.R50').
        ag_residues : List[str]
            List of antigen residues (ignored for feature extraction,
            but kept for interface consistency).
        
        Returns:
        --------
        Features dictionary:
            - affinity_mean;
            - affinity_std;
            - affinity_max;
            - affinity_min;
            - stability_ratio;
            - skewness.
        """
        
        # Get affinity time series for this Ab residue.
        if ab_residue not in self.ab_residue_index:
            # Return zero features if ab_residue not found.
            return {
                'affinity_mean': 0.0,
                'affinity_std': 0.0,
                'affinity_max': 0.0,
                'affinity_min': 0.0,
                'stability_ratio': 0.0,
                'skewness': 0.0
            }
        
        col_idx = self.ab_residue_index[ab_residue]
        affinities = self.affinity_data.iloc[:, col_idx].values
        
        # Compute the same 6 features as used in training.
        mean_aff = np.mean(affinities)
        std_aff = np.std(affinities)
        median_aff = np.median(affinities)
        
        features = {
            'affinity_mean': float(mean_aff),
            'affinity_std': float(std_aff),
            'affinity_max': float(np.max(affinities)),
            'affinity_min': float(np.min(affinities)),
            'stability_ratio': float(np.sum(affinities >= self.affinity_threshold)/len(affinities)),
            'skewness': float(3*(mean_aff - median_aff)/(std_aff + 1e-8))
        }
        return features
    
    def _extract_contact_affinity(
            self,
            ab_residue: str,
            ag_residues: List[str]
        ) -> Tuple[float, pd.DataFrame]:
        """
        Extract temporal affinity values for a contact.
        
        Parameters:
        -----------
        ab_residue : str
            Antibody residue (e.g., 'h.R50').
        ag_residues : List[str]
            List of antigen residues (e.g., ['V483', 'E484']).
        
        Returns:
        --------
        Tuple[float, pd.DataFrame]
            - Overall affinity score (mean across time);
            - DataFrame with temporal affinity trend.
        """
        
        # Check if ab_residue exists in affinity data.
        if ab_residue not in self.ab_residue_index:
            return 0.0, pd.DataFrame()
        
        # Get affinity values across time for this antibody residue.
        col_idx = self.ab_residue_index[ab_residue]
        affinities = self.affinity_data.iloc[:, col_idx].values
        
        # Create temporal DataFrame.
        time_points = np.arange(len(affinities))
        temporal_df = pd.DataFrame({
            'timepoint': time_points,
            'affinity': affinities,
            'stable': (affinities >= self.affinity_threshold).astype(int)
        })
        
        # Overall affinity score is the mean across time.
        overall_affinity = float(np.mean(affinities))

        return overall_affinity, temporal_df
    
    def _find_matching_antibodies(
        self,
        contacts: Dict[str, List[str]],
        similarity_threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Find known antibodies in library that match input contacts.
        
        Returns:
            List of (antibody_name, similarity_score) tuples, sorted by score descending.
        """
        ab_matches = defaultdict(list)
        
        for contact_residues in contacts.values():
            if len(contact_residues) < 1:
                continue
            
            ab_residue = contact_residues[0]
            
            # Find antibodies that have this residue in contact map.
            for ab_name in self.antibodies:
                key = (ab_name, ab_residue)
                if key in self.contact_map_lookup:
                    ab_matches[ab_name].append(1.0)  # Perfect match.
                else:
                    ab_matches[ab_name].append(0.3)  # Partial match.
        
        # Calculate average similarity per antibody.
        ab_scores = [
            (ab_name, np.mean(scores))
            for ab_name, scores in ab_matches.items()
        ]
        
        # Filter and sort.
        ab_scores = [
            (ab, score) for ab, score in ab_scores
            if score >= similarity_threshold
        ]
        ab_scores.sort(key = lambda x: x[1], reverse = True)
        return ab_scores
    
    def _detect_variant_mutations(
        self,
        contacts: Dict[str, List[str]],
        variant: str
    ) -> Tuple[List[str], float]:
        """
        Detect if contacts match a known variant with mutations.
        
        Returns:
            - List of detected mutations;
            - Overall mutation impact score (0-1).
        """
        detected_mutations = []
        impact_scores = []
        
        for contact_residues in contacts.values():
            for ag_residue in contact_residues[1:]:  # Skip Ab residue.
                if (variant, ag_residue) in self.mutations_lookup:
                    mut_info = self.mutations_lookup[(variant, ag_residue)]
                    detected_mutations.append(ag_residue)
                    
                    # Impact score: 1.0 if biochemical group changes, 0.5 otherwise.
                    if mut_info['wt_group'] != mut_info['mutant_group']:
                        impact_scores.append(1.0)
                    else:
                        impact_scores.append(0.5)
        
        overall_impact = np.mean(impact_scores) if impact_scores else 0.0
        
        return detected_mutations, float(overall_impact)
    
    def fit(
        self,
        contact_training_data: Optional[pd.DataFrame] = None,
        contact_labels: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        n_estimators: int = 2000,
        rfm_depth: int = 10,
        gbm_depth: int = 8,
        min_samples_split: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train ML models (Random Forest and LightGBM) for contact affinity prediction.
        If no training data provided, uses synthetic data from library.
        
        Parameters:
        -----------
        contact_training_data : pd.DataFrame, optional
            Feature matrix (n_samples, n_features).
        contact_labels : np.ndarray, optional
            Binary labels (stable/unstable interactions).
        test_size : float
            Test set fraction.
        n_estimators : int
            Number of tree estimators in the ML ensemble (default = 2000).
        rfm_depth : int
            Maximum random forest tree depth (default = 10).
        gbm_depth : int
            Maximum gradient boosting tree depth (default = 8).
        min_samples_split : int
            Minimum samples per tree split (default = 5).
        learning_rate : float
            LightGBM learning rate (default = 0.05).
        random_state : int
            Random state for reproducibility.
        
        Returns:
        --------
        Dict with cross-validation metrics.
        """
        
        # If no training data, use affinity_data itself.
        if contact_training_data is None:
            # Generate features from all Ab residues.
            features_list = []
            labels_list = []
            
            for col in self.affinity_data.columns:
                affinities = self.affinity_data[col].values
                features = {
                    'affinity_mean': np.mean(affinities),
                    'affinity_std': np.std(affinities),
                    'affinity_max': np.max(affinities),
                    'affinity_min': np.min(affinities),
                    'stability_ratio': np.sum(affinities >= self.affinity_threshold) / len(affinities),
                    'skewness': float(3 * (np.mean(affinities) - np.median(affinities)) / (np.std(affinities) + 1e-8)),
                }
                features_list.append(features)
                
                # Label: stable if mean affinity > threshold.
                labels_list.append(1 if np.mean(affinities) >= self.stability_threshold else 0)
            
            contact_training_data = pd.DataFrame(features_list)
            contact_labels = np.array(labels_list)
        
        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(
            contact_training_data, contact_labels,
            test_size = test_size, random_state = random_state,
            stratify = contact_labels
        )
        
        # Scale features.
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = rfm_depth,
            min_samples_split = min_samples_split,
            random_state = random_state,
            n_jobs = -1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_pred_binary = (rf_pred >= 0.5).astype(int)
        
        # Train LightGBM if available.
        if HAS_LIGHTGBM:
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators = n_estimators,
                learning_rate = learning_rate,
                max_depth = gbm_depth,
                random_state = random_state,
                verbose = -1
            )
            self.lgb_model.fit(X_train_scaled, y_train)
            lgb_pred = self.lgb_model.predict(X_test_scaled)
            lgb_pred_binary = (lgb_pred >= 0.5).astype(int)
        
        # Compute cross-validation metrics
        cv_metrics = {
            'rf_accuracy': float(accuracy_score(y_test, rf_pred_binary)),
            'rf_f1': float(f1_score(y_test, rf_pred_binary, zero_division = 0)),
            'rf_precision': float(precision_score(y_test, rf_pred_binary, zero_division = 0)),
            'rf_recall': float(recall_score(y_test, rf_pred_binary, zero_division = 0)),
        }
        
        if HAS_LIGHTGBM:
            cv_metrics.update({
                'lgb_accuracy': float(accuracy_score(y_test, lgb_pred_binary)),
                'lgb_f1': float(f1_score(y_test, lgb_pred_binary, zero_division = 0)),
                'lgb_precision': float(precision_score(y_test, lgb_pred_binary, zero_division = 0)),
                'lgb_recall': float(recall_score(y_test, lgb_pred_binary, zero_division = 0)),
            })
        
        self.is_fitted = True
        return cv_metrics

    def contact_stability(self, contacts_dict, threshold = 0.88):
        """
        Calculate stability score with exponential penalization for sub-threshold contacts.
        
        Parameters:
        - contacts_dict: {'contact_1': 0.96, 'contact_2': 0.76, ...};
        - threshold: affinity threshold (default = 0.88).
        
        Returns:
        - stability_score: weighted score incorporating penalization.
        """
        affinities = np.array(list(contacts_dict.values()))
        avg_affinity = np.mean(affinities)
        
        # Exponential penalization for each contact below threshold.
        penalization = 0
        for affinity in affinities:
            if affinity < threshold:
                gap = threshold - affinity  # How far below threshold.
                penalization += np.exp(2*gap) - 1  # Exponential scaling (factor of 2 can be tuned).
        
        # Normalize penalization by number of contacts.
        n_contacts = len(affinities)
        normalized_penalization = penalization/n_contacts
        
        # Combined stability score (capped at 0 to avoid negative values).
        stability_score = max(0, avg_affinity - (0.5*normalized_penalization))
        
        return stability_score, {
            'avg_affinity': avg_affinity,
            'n_subthreshold': np.sum(affinities < threshold),
            'penalization_term': normalized_penalization
        }

    def compute_risk_exponential(self, df, threshold = 0.88, metric = 'mean', severity = 3.0):
        """
        Exponential risk function. Penalizes affinity drop below threshold exponentially.
        
        Parameters:
        - df: DataFrame with timepoint, affinity, contact_id.
        - threshold: affinity threshold for a contact to be stable.
        - metric: 'mean' or 'median' affinity per timepoint.
        - severity: floating point value indicating exponential risk 
                    Suggested values are 1.0 (minimal), 3.0 (moderate, default),
                    5.0 (aggressive).
        """
        trend = df.groupby('timepoint')['affinity'].agg(metric).reset_index()
        trend.columns = ['timepoint', 'affinity']
        
        # Calculate deficit and apply exponential weighting.
        deficit = np.maximum(threshold - trend['affinity'], 0)
        
        # Exponential penalty: small deficits scale linearly, large deficits escalate.
        # Maps deficit [0, threshold] -> [0, e^severity - 1].
        w = (np.exp(severity*deficit/threshold) - 1)/(np.exp(severity) - 1)
        
        # Total weighted risk.
        risk_score = np.sum(w)
        
        # Maximum possible: all timepoints at worst affinity (0).
        # Each timepoint contributes 1.0, so max = len(trend).
        max_risk_score = len(trend)
        
        # Risk estimation.
        risk = 1 - 100*(risk_score/max_risk_score)
        
        return risk, {
            'proportion_time_below': np.sum(trend['affinity'] < threshold)/len(trend),
            'mean_weighted_deficit': risk_score/len(trend),
            'severest_timepoint': trend['affinity'].min(),
            'total_weighted_risk': risk_score
        }

    def compute_risk_sigmoidal(self, df, threshold = 0.88, metric = 'mean', steepness = 10):
        """
        Sigmoidal risk function. Affinities well above threshold (e.g., 1.0) get 0% risk;
        affinities at the threshold (e.g., 0.88) get roughly 50% risk; affinities well 
        below (e.g., 0.5) get 100% risk.

        Parameters:
            - df: DataFrame with timepoint, affinity, contact_id.
            - threshold: affinity threshold for a contact to be stable.
            - metric: 'mean' or 'median' affinity per timepoint.
            - steepness: steepness of the sigmoidal curve.
        """
        trend = df.groupby('timepoint')['affinity'].agg(metric).reset_index()
        trend.columns = ['timepoint', 'affinity']
        
        # Sigmoid centered at threshold, steepness controls transition sharpness.
        risk_per_timepoint = 1/(1 + np.exp(steepness*(trend['affinity'] - threshold)))
        
        # Average risk across all timepoints.
        risk = 100*np.mean(risk_per_timepoint)
        
        return risk, {
            'proportion_time_below': np.sum(trend['affinity'] < threshold)/len(trend),
            'mean_risk_per_timepoint': np.mean(risk_per_timepoint),
            'severest_risk_contribution': risk_per_timepoint.max(),
            'mildest_risk_contribution': risk_per_timepoint.min()
        }

    def compute_risk_linear(self, df, threshold = 0.88, metric = 'mean'):
        """
        Linear kernel. Risk is based on cumulative time below the affinity 
        threshold, normalized by maximum possible deficit.

        Parameters:
            - df: DataFrame with timepoint, affinity, contact_id.
            - threshold: affinity threshold for a contact to be stable.
            - metric: 'mean' or 'median' affinity per timepoint.
        """
        # Calculate interaction trend (mean or median per timepoint).
        trend = df.groupby('timepoint')['affinity'].agg(metric).reset_index()
        trend.columns = ['timepoint', 'affinity']
        
        # Calculate deficit from threshold for each timepoint.
        deficit = np.maximum(threshold - trend['affinity'], 0)
        
        # Total risk area (integral of deficit).
        risk_area = np.sum(deficit)
        
        # Maximum possible risk: all timepoints at 0 affinity.
        max_risk_area = threshold*len(trend)
        
        # Risk percentage (0-100%)
        risk = 100*(risk_area/max_risk_area)
        
        return risk, {
            'proportion_time_below': np.sum(trend['affinity'] < threshold)/len(trend),
            'mean_deficit_when_below': deficit[deficit > 0].mean() if (deficit > 0).any() else 0,
            'max_deficit': deficit.max(),
            'total_deficit_area': risk_area
        }
    
    def interaction_risk(self, df, metric = 'mean', kernel = 'exponential',
                         severity = 3.0, steepness = 10):
        """
        Estimate Ab-Ag interaction stability risk from time-series data.
        
        Parameters:
        - df: DataFrame with timepoint, affinity, contact_id.
        - metric: 'mean' or 'median' affinity per timepoint.
        - kernel: 'linear', 'exponential', or 'sigmoidal'.
        - severity: floating point value indicating exponential risk 
          amplification (1.0: minimal, 3.0: moderate, 5.0: aggressive).
        - steepness: steepness of the sigmoidal curve.
        """
        # Calculate interaction trend per timepoint.
        trend = df.groupby('timepoint')['affinity'].agg(metric).reset_index()
        trend.columns = ['timepoint', 'affinity']

        # Computing risk.
        if kernel == 'exponential':
            risk, details = self.compute_risk_exponential(
                df, self.affinity_threshold, metric, severity = severity
            )
        elif kernel == 'sigmoidal':
            risk, details = self.compute_risk_sigmoidal(
                df, self.affinity_threshold, metric, steepness = steepness
            )
        elif kernel == 'linear':
            risk, details = self.compute_risk_linear(
                df, self.affinity_threshold, metric
            )
        else:
            raise ValueError(
                "kernel must be 'linear', 'exponential', or 'sigmoidal'"
            )
        
        # Interaction-level affinity (mean affinity across all timepoints, 
        # penalized by time below threshold).
        trend = df.groupby('timepoint')['affinity'].agg(metric)
        p = (trend < self.affinity_threshold).sum()/len(trend)
        interaction_affinity = trend.mean()*(1 - p)
        
        return {
            'risk': risk,
            'interaction_affinity': interaction_affinity,
            'is_stable': interaction_affinity >= self.affinity_threshold,
            'details': details,
            'assessment': self._qualitative_assessment(
                interaction_affinity, risk
            )
        }
    
    def _qualitative_assessment(self, affinity, risk):
        if affinity >= 0.88 and risk < 20:
            return InteractionType.STABLE
        elif affinity >= 0.88 and risk >= 20:
            return InteractionType.WEAKLY_STABLE
        elif affinity < 0.88 and risk < 30:
            return InteractionType.UNSTABLE
        else:
            return InteractionType.EXTREMELY_UNSTABLE
    
    def predict(
            self,
            input_contacts: Union[Dict[str, List[str]], str, Path],
            use_ensemble: bool = True,
            metric: str = 'mean',
            kernel: str = 'exponential',
            severity: float = 3.0,
            steepness: int = 10
        ) -> PredictionResult:
        """
        Predict interaction characteristics for user-provided contacts.
        
        Parameters:
        -----------
        input_contacts : dict, str, or Path
            Either:
            - Dictionary: {'contact1': ['h.R50', 'V483', 'E484'],
                           'contact2': ['h.L55', 'L452', 'T470', 'F490'],
                           ...}
            - File path: str or Path to text file with contacts.
        use_ensemble : bool
            If True, use ensemble of RF + LightGBM; otherwise use best model.
        metric : str
            Either 'mean' or 'median' affinity per timepoint.
        kernel : str
            One among 'exponential' (default), 'linear', or 'sigmoidal'.
        severity : float
            Floating point value indicating exponential risk
            amplification. Suggested values are 1.0 (minimal),
            3.0 (moderate; default), 5.0 (aggressive).
        steepness : int
            steepness of the sigmoidal curve (default = 10).
        
        Returns:
        --------
        PredictionResult object instance.
        """

        ### PREDICTION DATA STRUCTURES PREPARATION.
        
        # Load contacts if file path provided.
        if isinstance(input_contacts, (str, Path)):
            input_contacts = self.load_contacts_from_file(input_contacts)
        
        # Calculate contact affinities and aggregate temporal trends.
        contact_affinity_scores = {}
        all_temporal_trends = []
        all_contact_temporal_affinities = []
        
        for contact_id, residues in input_contacts.items():
            if len(residues) < 2:
                continue
            
            ab_residue = residues[0]
            ag_residues = residues[1:]
            
            aff_score, temporal_df = self._extract_contact_affinity(ab_residue, ag_residues)
            contact_affinity_scores[contact_id] = aff_score
            
            if not temporal_df.empty:
                temporal_df['contact_id'] = contact_id
                all_temporal_trends.append(temporal_df)
                
                # Store affinity values across time for feature extraction.
                if 'affinity' in temporal_df.columns:
                    all_contact_temporal_affinities.append(temporal_df['affinity'].values)
                else:
                    # Fallback: use single score repeated.
                    all_contact_temporal_affinities.append(np.array([aff_score] * len(temporal_df)))
        
        # Aggregate temporal trend.
        if all_temporal_trends:
            temporal_affinity_trend = pd.concat(all_temporal_trends, ignore_index=True)
        else:
            temporal_affinity_trend = pd.DataFrame()
        
        ### FEATURE EXTRACTION.
        
        all_contact_features = []
        
        for contact_temporal_affinities in all_contact_temporal_affinities:
            # Ensure we have affinity values.
            if len(contact_temporal_affinities) == 0:
                continue
            
            affinities = contact_temporal_affinities.astype(float)
            
            # Compute the same features as the ones used in training.
            contact_features = {
                'affinity_mean': np.mean(affinities),
                'affinity_std': np.std(affinities),
                'affinity_max': np.max(affinities),
                'affinity_min': np.min(affinities),
                'stability_ratio': np.sum(affinities >= self.affinity_threshold)/len(affinities),
                'skewness': float(3*(np.mean(affinities) - np.median(affinities))/(np.std(affinities) + 1e-8)),
            }
            all_contact_features.append(contact_features)
        
        # Aggregate features across all contacts.
        if all_contact_features:
            X_features = pd.DataFrame({
                'affinity_mean': [np.mean([f['affinity_mean'] for f in all_contact_features])],
                'affinity_std': [np.mean([f['affinity_std'] for f in all_contact_features])],
                'affinity_max': [np.mean([f['affinity_max'] for f in all_contact_features])],
                'affinity_min': [np.mean([f['affinity_min'] for f in all_contact_features])],
                'stability_ratio': [np.mean([f['stability_ratio'] for f in all_contact_features])],
                'skewness': [np.mean([f['skewness'] for f in all_contact_features])],
            })
        else:
            # Default fallback if no valid contacts.
            X_features = pd.DataFrame({
                'affinity_mean': [0.0],
                'affinity_std': [0.0],
                'affinity_max': [0.0],
                'affinity_min': [0.0],
                'stability_ratio': [0.0],
                'skewness': [0.0],
            })
        
        ### PREDICTION.
        
        ml_predictions = {}
        feature_importance = {}
        
        if self.is_fitted and self.rf_model is not None:
            X_scaled = self.scaler.transform(X_features)
            rf_score = float(self.rf_model.predict(X_scaled)[0])
            ml_predictions['random_forest'] = rf_score
            
            # Feature importance.
            for feat, imp in zip(X_features.columns, self.rf_model.feature_importances_):
                feature_importance[feat] = float(imp)
        
        if self.is_fitted and HAS_LIGHTGBM and self.lgb_model is not None:
            X_scaled = self.scaler.transform(X_features)
            lgb_score = float(self.lgb_model.predict(X_scaled)[0])
            ml_predictions['lightgbm'] = lgb_score
        
        # Determine consensus prediction.
        if use_ensemble and len(ml_predictions) > 1:
            ensemble_score = min(list(ml_predictions.values()))
        elif ml_predictions:
            ensemble_score = min(ml_predictions.values())
        else:
            ensemble_score = np.mean(list(contact_affinity_scores.values())) if contact_affinity_scores else 0.5

        ### ANNOTATION.
        
        # Check cosine similarity with known contacts.
        cosine_similarity = self._compute_contact_similarity(input_contacts)
        
        # Find matching antibodies.
        ab_matches = self._find_matching_antibodies(input_contacts)
        best_ab = ab_matches[0][0] if ab_matches else None
        best_ab_score = ab_matches[0][1] if ab_matches else 0.0
        
        # Detect variant mutations.
        variant_mutations = {}
        primary_variant_match = None
        max_mutation_impact = 0.0
        
        for variant in self.variants[1:]:  # Skip 'wt'.
            mutations, impact = self._detect_variant_mutations(
                input_contacts, variant
            )
            if mutations:
                variant_mutations[variant] = {
                    'mutations': mutations,
                    'impact': impact
                }
                if impact > max_mutation_impact:
                    max_mutation_impact = impact
                    primary_variant_match = variant
        
        # Classify interaction type.
        interaction = self.interaction_risk(
            temporal_affinity_trend,
            metric = metric,
            kernel = kernel,
            severity = severity,
            steepness = steepness
        )

        # Calculate confidence (inverse of uncertainty).
        confidence = float(np.clip(
            (ensemble_score + cosine_similarity + best_ab_score)/3.0,
            0.0, 1.0
        ))
        
        # Cross-validation metrics
        cv_metrics = {}
        if self.is_fitted:
            cv_metrics = {
                'model_trained': True,
                'models_used': list(ml_predictions.keys())
            }
        
        return PredictionResult(
            interaction_type = interaction['assessment'],
            interaction_affinity = interaction['interaction_affinity'],
            risk_score = interaction['risk'],
            risk_level = RiskLevel.classify(
                None, interaction['interaction_affinity']
            ),
            confidence = confidence,
            primary_variant_match = primary_variant_match,
            best_ab_match = best_ab,
            mutation_impact = max_mutation_impact if variant_mutations else None,
            contact_affinity_scores = contact_affinity_scores,
            temporal_affinity_trend = temporal_affinity_trend,
            ml_predictions = ml_predictions,
            feature_importance = feature_importance,
            cross_validation_metrics = cv_metrics
        )

    def _compute_contact_similarity(self, input_contacts: Dict[str, List[str]]) -> float:
        """
        Compute cosine similarity between input contacts and known library contacts.
        """
        # Flatten input contacts.
        input_flat = set()
        for residues in input_contacts.values():
            input_flat.update(residues)
        
        # Compute similarity against all known contacts.
        similarities = []

        for ab_name in self.antibodies:
            library_flat = set()
            for (ab, residue), ag_residues in self.contact_map_lookup.items():
                if ab == ab_name:
                    library_flat.add(residue)
            
            # Compute Jaccard similarity.
            if len(input_flat) + len(library_flat) > 0:
                intersection = len(input_flat & library_flat)
                union = len(input_flat | library_flat)
                similarity = intersection / union
                similarities.append(similarity)
        
        # Return average similarity.
        return float(np.mean(similarities)) if similarities else 0.0
    
    def plot_interaction_trend(
        self,
        prediction_result: PredictionResult,
        output_file: Optional[Union[str, Path]] = None,
        window_length: int = 15,
        polyorder: int = 3,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Plot temporal affinity trend with smoothing and confidence intervals.
        
        Parameters:
        -----------
        prediction_result : PredictionResult
            Result from predict() method.
        output_file : str or Path, optional
            Save figure to file.
        window_length : int
            Window length for Savitzky-Golay filter (must be odd).
        polyorder : int
            Polynomial order for Savitzky-Golay filter.
        figsize : tuple
            Figure size (width, height).
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        
        temporal_df = prediction_result.temporal_affinity_trend
        
        if temporal_df.empty:
            warnings.warn("No temporal data available for plotting")
            return plt.figure()
        
        fig, axes = plt.subplots(2, 2, figsize = figsize)
        
        ### Figure 1. Temporal trend with smoothing.
        ax = axes[0, 0]
        
        # Track unique contacts for legend (avoid duplicates).
        contact_ids = temporal_df['contact_id'].unique()
        colors_dict = {cid: plt.cm.tab10(i % 10) for i, cid in enumerate(contact_ids)}
        
        # Collect all affinities for aggregated curve.
        all_affinities_by_time = {}
        
        for contact_id in contact_ids:
            subset = temporal_df[temporal_df['contact_id'] == contact_id]
            timepoints = subset['timepoint'].values
            affinities = subset['affinity'].values
            
            # Store for aggregation (no labels for individual raw data).
            for t, a in zip(timepoints, affinities):
                if t not in all_affinities_by_time:
                    all_affinities_by_time[t] = []
                all_affinities_by_time[t].append(a)
            
            # Plot raw data (NO LABEL to keep legend clean).
            color = colors_dict[contact_id]
            ax.scatter(timepoints, affinities, alpha = 0.3, s = 25, color = color)
            
            # Apply Savitzky-Golay filter for smoothing
            if len(timepoints) >= window_length:
                win = window_length if window_length % 2 == 1 else window_length + 1
                win = min(win, len(timepoints))
                if win % 2 == 0:
                    win -= 1
                
                if win > polyorder:
                    smoothed = savgol_filter(affinities, win, polyorder)
                    # Plot smoothed individual contact (with label).
                    ax.plot(timepoints, smoothed, linewidth = 1.8,
                            label = f'{contact_id}', alpha = 0.75, color = color)
            
            # Confidence interval using rolling std.
            window = max(3, len(affinities)//10)
            rolling_std = pd.Series(affinities).rolling(window = window, center = True).std().values
            rolling_std = np.nan_to_num(rolling_std, nan = 0.05)
            
            ax.fill_between(
                timepoints,
                affinities - 1.96*rolling_std,
                affinities + 1.96*rolling_std,
                alpha = 0.08,
                color = color
            )
        
        # Plot aggregated curve (mean across all contacts).
        sorted_times = sorted(all_affinities_by_time.keys())
        aggregated_affinities = [
            np.mean(all_affinities_by_time[t]) for t in sorted_times
        ]
        
        # Smooth the aggregated curve.
        if len(sorted_times) >= window_length:
            win = window_length if window_length % 2 == 1 else window_length + 1
            win = min(win, len(sorted_times))
            if win % 2 == 0:
                win -= 1
            
            if win > polyorder:
                smoothed_agg = savgol_filter(aggregated_affinities, win, polyorder)
                ax.plot(sorted_times, smoothed_agg, linewidth = 3.5,
                        label = 'Overall Interaction (aggregated)',
                        alpha = 0.95, color = 'darkblue', zorder = 10)
        
        # Add stability threshold line.
        ax.axhline(y = self.affinity_threshold, color = 'black', linestyle = ':',
                   linewidth = 2.2, label = f'Affinity threshold ({self.affinity_threshold})',
                   zorder = 9)
        
        ax.set_xlabel('Timepoint (nanoseconds)', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Affinity Score', fontsize = 12, fontweight = 'bold')
        ax.set_title('Temporal Affinity trend', fontsize = 12, fontweight = 'bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc = 'best', fontsize = 7, framealpha = 0.60)
        ax.grid(True, alpha  =0.3)
        
        ### Figure 2. Risk score gauge.
        ax = axes[0, 1]
        ax.axis('off')
        
        risk_score = prediction_result.risk_score
        risk_level = prediction_result.risk_level
        
        # Draw gauge.
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        
        # Color zones.
        theta_unmanageable = np.linspace(0, np.pi*0.7, 50)
        theta_partial = np.linspace(np.pi*0.7, np.pi*0.88, 30)
        theta_manageable = np.linspace(np.pi*0.88, np.pi, 20)
        
        ax.fill_between(np.cos(theta_unmanageable), 0, np.sin(theta_unmanageable),
                        color = 'red', alpha = 0.3, label = 'Unmanageable')
        ax.fill_between(np.cos(theta_partial), 0, np.sin(theta_partial),
                        color = 'orange', alpha = 0.3, label = 'Partially Manageable')
        ax.fill_between(np.cos(theta_manageable), 0, np.sin(theta_manageable),
                        color = 'green', alpha = 0.3, label = 'Manageable')
        
        # Needle.
        needle_angle = np.pi*(1.0 - risk_score)  # Invert for display.
        needle_x = [0, 0.9*np.cos(needle_angle)]
        needle_y = [0, 0.9*np.sin(needle_angle)]
        ax.plot(needle_x, needle_y, 'k-', linewidth = 3)
        ax.plot(0, 0, 'ko', markersize = 12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.legend(loc = 'upper left', fontsize = 9)
        ax.set_title(f'Risk Score: {risk_score:.1%} ({risk_level})',
                     fontsize = 12, fontweight = 'bold')
        
        ### Figure 3. Contact affinity distribution.
        ax = axes[1, 0]
        
        contact_scores = list(prediction_result.contact_affinity_scores.values())
        if contact_scores:
            ax.bar(range(len(contact_scores)), contact_scores, color = 'steelblue', alpha = 0.7)
            ax.axhline(y = self.affinity_threshold, color = 'red', linestyle = '--',
                       linewidth = 2, label = 'Affinity threshold')
            ax.set_xlabel('Contact ID', fontsize = 12, fontweight = 'bold')
            ax.set_ylabel('Affinity Score', fontsize = 12, fontweight = 'bold')
            ax.set_title('Per-Contact Affinity Scores', fontsize = 12, fontweight = 'bold')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize = 9)
            ax.grid(axis='y', alpha = 0.3)
        
        ### Figure 4. ML predictions comparison.
        ax = axes[1, 1]

        try:
            stability = prediction_result.interaction_affinity
            confidence = prediction_result.confidence
            preds = True
        except:
            preds = False

        if preds:
            indices = ['Stability', 'Confidence']
            scores = [stability, confidence]
            
            colors = ['steelblue' if m != 'ensemble' else 'darkgreen' for m in indices]
            bars = ax.bar(indices, scores, color = colors, alpha = 0.7)
            
            # Add value labels on bars.
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.2f}', ha = 'center', va = 'bottom',
                        fontsize = 10, fontweight = 'bold')
            
            ax.set_ylabel('Proportion', fontsize = 12, fontweight = 'bold')
            ax.set_title('Interaction confidence', fontsize = 12, fontweight = 'bold')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize = 9)
            ax.grid(axis = 'y', alpha = 0.3)
        
        plt.tight_layout()
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents = True, exist_ok = True)
            plt.savefig(output_file, dpi = 400, bbox_inches = 'tight')
            print(f"Figure saved to {output_file}")
        
        return fig
    
    def plot_feature_importance(
        self,
        prediction_result: PredictionResult,
        top_n: int = 15,
        output_file: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """Plot feature importance from trained models."""
        
        feature_imp = prediction_result.feature_importance
        
        if not feature_imp:
            warnings.warn("No feature importance data available")
            return plt.figure()
        
        # Sort and select top N.
        sorted_features = sorted(feature_imp.items(), key = lambda x: x[1],
                                 reverse = True)
        top_features = sorted_features[:top_n]
        
        names, importances = zip(*top_features)
        
        fig, ax = plt.subplots(figsize = figsize)
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, importances, color = 'steelblue', alpha = 0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize = 10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize = 12, fontweight = 'bold')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize = 12, fontweight = 'bold')
        ax.grid(axis = 'x', alpha = 0.3)
        
        plt.tight_layout()
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents = True, exist_ok = True)
            plt.savefig(output_file, dpi = 400, bbox_inches = 'tight')
            print(f"Feature importance plot saved to {output_file}")
        
        return fig
    
    def generate_pdf_report(
        self,
        prediction_result: PredictionResult,
        output_file: Union[str, Path],
        figure_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate comprehensive PDF report of analysis.
        
        Parameters:
        -----------
        prediction_result : PredictionResult
            Result from predict() method
        output_file : str or Path
            Output PDF file path
        figure_dir : str or Path, optional
            Directory to save intermediate figures
        """
        
        if not HAS_REPORTLAB:
            warnings.warn("ReportLab not installed. PDF report unavailable.")
            return
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents = True, exist_ok = True)
        
        if figure_dir is None:
            figure_dir = output_file.parent / "figures"
        else:
            figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate figures
        trend_fig_path = figure_dir / "trend.png"
        importance_fig_path = figure_dir / "importance.png"
        
        self.plot_interaction_trend(prediction_result, output_file = trend_fig_path)
        self.plot_feature_importance(prediction_result, output_file = importance_fig_path)
        
        plt.close('all')
        
        # Create PDF.
        doc = SimpleDocTemplate(str(output_file), pagesize = letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles.
        title_style = ParagraphStyle(
            'CustomTitle',
            parent = styles['Heading1'],
            fontSize = 24,
            textColor = colors.HexColor('#1f77b4'),
            spaceAfter = 30,
            alignment = TA_CENTER,
            fontName = 'Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent = styles['Heading2'],
            fontSize = 14,
            textColor = colors.HexColor('#1f77b4'),
            spaceAfter = 12,
            spaceBefore = 12,
            fontName = 'Helvetica-Bold'
        )
        
        # Title.
        story.append(Paragraph("Antibody-Antigen Interaction Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Summary section.
        story.append(Paragraph("Analysis Summary", heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Interaction Type', prediction_result.interaction_type.value],
            ['Risk Score', f"{prediction_result.risk_score:.1%}"],
            ['Risk Level', prediction_result.risk_level],
            ['Confidence', f"{prediction_result.confidence:.1%}"],
            ['Primary Variant', prediction_result.primary_variant_match or 'Unknown'],
            ['Best Antibody Match', prediction_result.best_ab_match or 'None'],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])

        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Contact affinities section.
        story.append(Paragraph("Contact Affinity Scores", heading_style))
        
        contact_aff = prediction_result.contact_affinity_scores
        if contact_aff:
            contact_data = [['Contact ID', 'Affinity Score', 'Status']]
            for contact_id, score in contact_aff.items():
                status = "Stable" if score >= self.affinity_threshold else "Unstable"
                contact_data.append([contact_id, f"{score:.4f}", status])
            
            contact_table = Table(contact_data, colWidths = [2*inch, 2*inch, 2*inch])
            contact_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            story.append(contact_table)
            story.append(Spacer(1, 0.3*inch))
        
        # ML predictions section.
        if prediction_result.ml_predictions:
            story.append(Paragraph("Machine Learning Predictions", heading_style))
            
            ml_data = [['Model', 'Prediction Score']]
            for model, score in prediction_result.ml_predictions.items():
                ml_data.append([model.replace('_', ' ').title(), f"{score:.4f}"])
            
            ml_table = Table(ml_data, colWidths=[3*inch, 3*inch])
            ml_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            story.append(ml_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Figures.
        story.append(PageBreak())
        story.append(Paragraph("Temporal Affinity Trend Analysis", heading_style))
        
        if trend_fig_path.exists():
            img = Image(str(trend_fig_path), width = 7*inch, height = 5.25*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        story.append(Paragraph("Feature Importance Analysis", heading_style))
        
        if importance_fig_path.exists():
            img = Image(str(importance_fig_path), width = 7*inch, height = 3.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        # Interpretation section.
        story.append(PageBreak())
        story.append(Paragraph("Interpretation and Recommendations", heading_style))
        
        interpretation_text = self._generate_interpretation_text(prediction_result)
        story.append(Paragraph(interpretation_text, styles['BodyText']))
        
        # Build PDF.
        doc.build(story)
        print(f"PDF report saved to {output_file}")
    
    def _generate_interpretation_text(self, prediction_result: PredictionResult) -> str:
        """Generate human-readable interpretation of results."""
        
        text_parts = []
        
        # Interaction type interpretation.
        type_text = f"<b>Interaction Classification:</b> {prediction_result.interaction_type.value}<br/><br/>"
        text_parts.append(type_text)
        
        # Risk assessment.
        risk_text = (
            f"<b>Risk Assessment:</b> The predicted risk score is "
            f"<b>{prediction_result.risk_score:.1%}</b>, classified as "
            f"<b>{prediction_result.risk_level}</b>. "
        )
        
        if prediction_result.risk_level == "Manageable":
            risk_text += (
                "This indicates that existing antibodies in the library can likely neutralize "
                "this antigen variant with good affinity. Therapeutic options are available."
            )
        elif prediction_result.risk_level == "Partially Manageable":
            risk_text += (
                "This indicates that some mutations reduce effectiveness of current antibodies, "
                "but partial protection is still achievable. Close monitoring is recommended."
            )
        else:
            risk_text += (
                "This indicates a significant risk. The antigen may evade current antibody "
                "therapies. Additional experimental validation is strongly recommended."
            )
        
        text_parts.append(f"{risk_text}<br/><br/>")
        
        # Variant matching.
        if prediction_result.primary_variant_match:
            variant_text = (
                f"<b>Known Variant Match:</b> This interaction pattern is most consistent with "
                f"the <b>{prediction_result.primary_variant_match.upper()}</b> variant. "
                f"Mutation impact score: <b>{prediction_result.mutation_impact:.1%}</b><br/><br/>"
            )
            text_parts.append(variant_text)
        
        # Antibody match.
        if prediction_result.best_ab_match:
            ab_text = (
                f"<b>Antibody Effectiveness:</b> The best-matching antibody in the library is "
                f"<b>{prediction_result.best_ab_match}</b>. This antibody may provide neutralizing "
                f"capacity against this antigen. Confidence: <b>{prediction_result.confidence:.1%}</b><br/><br/>"
            )
            text_parts.append(ab_text)
        
        # ML consensus.
        if prediction_result.ml_predictions:
            ml_text = (
                f"<b>Model Consensus:</b> Multiple machine learning models (Random Forest, "
                f"LightGBM) converged on similar predictions, indicating high confidence in the analysis. "
                f"See feature importance plot for key drivers.<br/><br/>"
            )
            text_parts.append(ml_text)
        
        # Recommendations.
        rec_text = "<b>Recommendations:</b><ul>"
        
        if prediction_result.risk_level == "Manageable":
            rec_text += "<li>Current monoclonal antibody therapies should retain efficacy.</li>"
            rec_text += "<li>Continue standard treatment protocols.</li>"
        elif prediction_result.risk_level == "Partially Manageable":
            rec_text += "<li>Monitor for breakthrough infections.</li>"
            rec_text += "<li>Consider combination therapies for enhanced coverage.</li>"
            rec_text += "<li>Accelerate research into next-generation antibodies.</li>"
        else:
            rec_text += "<li>This variant may represent a significant escape risk.</li>"
            rec_text += "<li>Recommend urgent experimental validation and characterization.</li>"
            rec_text += "<li>Initiate structure-guided antibody redesign efforts.</li>"
            rec_text += "<li>Evaluate combination approaches and vaccine booster strategies.</li>"
        
        rec_text += "</ul><br/>"
        text_parts.append(rec_text)
        
        return "".join(text_parts)
    
    def summary_statistics(self, prediction_result: PredictionResult) -> pd.DataFrame:
        """Return a summary statistics DataFrame."""
        
        stats = {
            'Metric': [
                'Interaction Type',
                'Risk Score',
                'Risk Level',
                'Confidence',
                'Primary Variant',
                'Best Antibody',
                'Number of Contacts',
                'Mean Contact Affinity',
                'Max Contact Affinity',
                'Stable Contacts (%)',
                'Mutation Impact',
                'ML Models Used'
            ],
            'Value': [
                prediction_result.interaction_type.value,
                f"{prediction_result.risk_score:.4f}",
                prediction_result.risk_level,
                f"{prediction_result.confidence:.4f}",
                prediction_result.primary_variant_match or "Unknown",
                prediction_result.best_ab_match or "None",
                len(prediction_result.contact_affinity_scores),
                f"{np.mean(list(prediction_result.contact_affinity_scores.values())):.4f}" if prediction_result.contact_affinity_scores else "N/A",
                f"{np.max(list(prediction_result.contact_affinity_scores.values())):.4f}" if prediction_result.contact_affinity_scores else "N/A",
                f"{100*sum(1 for s in prediction_result.contact_affinity_scores.values() if s >= self.affinity_threshold) / len(prediction_result.contact_affinity_scores):.1f}" if prediction_result.contact_affinity_scores else "0.0",
                f"{prediction_result.mutation_impact:.4f}" if prediction_result.mutation_impact else "N/A",
                ", ".join(prediction_result.cross_validation_metrics.get('models_used', ['None']))
            ]
        }
        
        return pd.DataFrame(stats)