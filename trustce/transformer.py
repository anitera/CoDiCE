import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import eigh
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class Transformer(object):
    def __init__(self, dataset, config):
        self.continuous_features_transformers = self.get_cont_features_transformers(dataset, config)
        self.categorical_features_transformers = self.get_cat_features_transformers(dataset, config)
        if config.get_config_value("cfsearch")["continuous_distance"]["type"]=="diffusion":
            dataset.data[dataset.continuous_features_list] = (dataset.data[dataset.continuous_features_list] - dataset.data[dataset.continuous_features_list].mean()) / dataset.data[dataset.continuous_features_list].std()
            self.normlaize_cont_dataset_numpy = dataset.data[dataset.continuous_features_list].to_numpy()
            self.k_neighbors = config.get_config_value("cfsearch")["continuous_distance"]["diffusion_params"]["k_neighbors"]
            self.alpha = config.get_config_value("cfsearch")["continuous_distance"]["diffusion_params"]["alpha"]
            self.diffusion_map, self.local_scale, self.eigenvalues, self.eigenvectors = self.custom_diff_map(k_neighbors=self.k_neighbors, 
                                                      alpha=self.alpha)
        self.mads = self._get_mads(dataset)

    def normalize_instance(self, instance):
        """Return normalize instance"""
        instance.normalized = True
        for feature_name, feature in instance.features.items():
            if feature_name in self.continuous_features_transformers:
                feature.value = self.continuous_features_transformers[feature_name].normalize_cont_value(feature.value) #TODO check for over-assignment
            elif feature_name in self.categorical_features_transformers:
                feature.value = self.categorical_features_transformers[feature_name].normalize_cat_value(feature.value)
            else:
                raise ValueError("Feature name is not in continuous or categorical features list")
        

    def get_cont_features_transformers(self, dataset, config):
        return {feature_name: FeatureTransformer(dataset, config, feature_name) for feature_name in dataset.continuous_features_list}

    def get_cat_features_transformers(self, dataset, config):
        return {feature_name: FeatureTransformer(dataset, config, feature_name) for feature_name in dataset.categorical_features_list}
    
    def get_cont_transformers_length(self):
        return len(self.continuous_features_transformers)
    
    def get_cat_transformers_length(self):
        return len(self.categorical_features_transformers)
    
    def denormalize_instance(self, instance):
        """Return denormalized instance"""
        instance.normalized = False
        for feature_name, feature in instance.features.items():
            if feature_name in self.continuous_features_transformers:
                feature.value = self.continuous_features_transformers[feature_name].denormalize_cont_value(feature.value) #TODO check for over-assignment
            elif feature_name in self.categorical_features_transformers:
                feature.value = self.categorical_features_transformers[feature_name].denormalize_cat_value(feature.value)
            else:
                raise ValueError("Feature name is not in continuous or categorical features list")

    def _get_diffusion_map(self, dataset, kernel_size=6, n_eigenvecs=3):
        """Calulate diffusion map fo coninuous features with gaussian kernel"""
        from pydiffmap import diffusion_map
        from pydiffmap import kernel

        # Normalize data
        dataset.data[dataset.continuous_features_list] = (dataset.data[dataset.continuous_features_list] - dataset.data[dataset.continuous_features_list].mean()) / dataset.data[dataset.continuous_features_list].std()

        # Computer median pairwise distance for kernel_size epsilon reference. 
        distances = distance_matrix(dataset.data[dataset.continuous_features_list].to_numpy(), dataset.data[dataset.continuous_features_list].to_numpy())
        kernel_size_estimate = np.median(distances)
        print("Kernel size estimate: ", kernel_size_estimate)

        # Estimate eigenvalues
        # Construct the affinity matrix using the Gaussian kernel
        affinity_matrix = np.exp(-distances**2 / (2 * kernel_size_estimate**2))
        shape_affinity_matrix = affinity_matrix.shape

        # Compute the diagonal matrix D (degree matrix)
        D = np.diag(1.0 / np.sqrt(np.sum(affinity_matrix, axis=1)))

        # Compute the normalized graph Laplacian (or diffusion matrix)
        L = D @ affinity_matrix @ D

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)

        # Since the eigenvalues are returned in ascending order, we'll reverse them to have the largest eigenvalues first
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        eigenvalues[:10]  # Displaying the top 10 eigenvalues
        print("Top eigenvalues: ", eigenvalues[:10]) # look for significant drop in eigenvalues

        # Get only continuous features
        shape = dataset.data[dataset.continuous_features_list].to_numpy()

        max_number_eigenvalues = (len(shape) - 1)//2

        kernel_gaussian = kernel.Kernel(kernel_type='gaussian', k=kernel_size, neighbor_params={'n_jobs': -1, 'algorithm': 'ball_tree'})
        diff_map = diffusion_map.DiffusionMap(kernel_gaussian, n_evecs=max_number_eigenvalues)
        # Fit the whole data. I should write my own diffusion map class to fit only on training data to debug where the problem is
        diff_map.fit(shape)

        return diff_map

    def custom_diff_map(self, k_neighbors=6, alpha=1.0):
        """Calculate diffusion map for continous features with number of neighbors as parameter"""
        # Normalize data
        dist_matrix = distance_matrix(self.normlaize_cont_dataset_numpy, self.normlaize_cont_dataset_numpy)
        # Self-tuning kernel
        local_scale = np.sort(dist_matrix, axis=1)[:, k_neighbors]
        local_scale_matrix = local_scale[:, np.newaxis] * local_scale
        affinity_matrix = np.exp(-dist_matrix ** 2 / local_scale_matrix)

        # Row normalize the affinity matrix
        row_sums = affinity_matrix.sum(axis=1)
        transition_matrix = affinity_matrix / row_sums[:, np.newaxis]

        # Compute all eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(transition_matrix.T)
        eigenvalues = np.real(eigenvalues[::-1])  # Reversing to get descending order
        eigenvectors = np.real(eigenvectors[:, ::-1])  # Reversing to match eigenvalues

        # Create the full-dimensional diffusion map
        diffusion_map = eigenvectors * eigenvalues**alpha
        return diffusion_map, local_scale, eigenvalues, eigenvectors


    def _get_mads(self, dataset):
        """Computes Median Absolute Deviation of features."""
        mads = {}
        # For all continuous gfeatures
        for feature in self.continuous_features_transformers:
            mads[feature] = np.median(abs(dataset.data[feature].values - np.median(dataset.data[feature].values)))
            if mads[feature] <= 0:
                mads[feature] = 1.0
                #if display_warnings:
                print(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)
        return mads


class FeatureTransformer(object):
    """
    Store original range of data, normalized range of data and normalization type
    """
    def __init__(self, dataset, config, feature_name):
        self.feature_name = feature_name
        if self.feature_name in dataset.continuous_features_list:
            self.min, self.max, self.mean, self.std, self.median = self.calculate_statistics(dataset)
            self.norm_type = config.get_config_value("model")["continuous_features_normalization"]
            if self.norm_type == "minmax":
                self.normalize_cont_value = lambda x: (x - self.min) / (self.max - self.min)
                self.denormalize_cont_value = lambda x: x * (self.max - self.min) + self.min
            elif self.norm_type == "standard":
                self.normalize_cont_value = lambda x: (x - self.mean) / self.std
                self.denormalize_cont_value = lambda x: x * self.std + self.mean
            elif self.norm_type == "none":
                self.normalize_cont_value = lambda x: x
                self.denormalize_cont_value = lambda x: x

            self.original_range = self.get_from_dataset(dataset)
            self.normalized_range = self.get_normalized_range(dataset)
        elif self.feature_name in dataset.categorical_features_list:
            self.original_range = self.get_feature_choice(dataset)
            self.enc_type = config.get_config_value("model")["categorical_features_encoding"] #TODO this could be done better with OOP
            if self.enc_type == "onehot":
                self._initialize_onehot_encoder(dataset)
                self.normalize_cat_value = self._onehot_enc
                self.denormalize_cat_value = self._onehot_dec
            elif self.enc_type == "ordinal":
                self.normalize_cat_value = self._ordinal_enc
                self.denormalize_cat_value = self._ordinal_dec
            elif self.enc_type == "frequency":
                self.normalize_cat_value = self._freq_enc
                self.denormalize_cat_value = self._freq_dec
            elif self.enc_type == "label_encoder":
                self._initialize_label_encoder(dataset)
                self.normalize_cat_value = self._label_enc
                self.denormalize_cat_value = self._label_dec
            elif self.enc_type == "none":
                # Probably should raise the warning that categorical feature is not encoded
                self.normalize_cat_value = lambda x: x
                self.denormalize_cat_value = lambda x: x
            else:
                raise ValueError("En_label_enccoding type is not supported")
            self.normalized_range = self.apply_encoding(dataset)
        else:
            raise ValueError("Feature name is not in continuous or categorical features list")

    def calculate_statistics(self, dataset):
        return dataset.data[self.feature_name].agg(["min", "max", "mean", "std", "median"])

    def get_from_dataset(self, dataset):
        aggregated_values = dataset.data[self.feature_name].agg(["min", "max"])
        return [aggregated_values["min"], aggregated_values["max"]]
    
    def get_feature_choice(self, dataset):
        return dataset.data[self.feature_name].unique()
    
    def apply_encoding(self, dataset):
        """Return feature range for categorical features"""
        if self.enc_type == "onehot":
            return [0,1]
        elif self.enc_type == "ordinal":
            number_categories = len(dataset.data[self.feature_name].unique())
            return [0, number_categories-1]
        elif self.enc_type == "label_encoder":
            return [0, len(dataset.data[self.feature_name].unique())-1]
        elif self.enc_type == "frequency":
            raise NotImplementedError
        else:
            raise ValueError("Encoding type is not supported")
    
    def get_normalized_range(self, dataset):
        if self.norm_type == "minmax":
            return [0,1]
        elif self.norm_type == "standard":
            # Calculate std for dataset[feature_name]
            standartised = (dataset.data[self.feature_name] - self.mean) / self.std
            return [standartised.min(), standartised.max()]
        elif self.norm_type == "none":
            return self.original_range
        else:
            raise ValueError("Normalization type is not supported")

    # def normalize_cont_value(self, config, original_value):
    #     if config.continuous_features["normalization"] == "minmax":
    #         return (original_value - self.min) / (self.max - self.min)
    #     elif config.continuous_features["normalization"] == "standart":
    #         return (original_value - self.mean) / self.std

    # def denormalize_cont_value(self, config, normalized_value):
    #     if config.continuous_features["normalization"] == "minmax":
    #         return normalized_value * (self.max - self.min) + self.min
    #     elif config.continuous_features["normalization"] == "standart":
    #         return normalized_value * self.std + self.mean
    
    # def normalize_cat_value(self, config, original_value):
    #     """The ordinal encoding looks weird, todo check it later"""
    #     if config.categorical_features["encoding"] == "onehot":
    #         return 1
    #     elif config.categorical_features["encoding"] == "ordinal":
    #         return original_value
    #     elif config.categorical_features["encoding"] == "frequency":
    #         raise NotImplementedError
        
    # def denormalize_cat_value(self, config, normalized_value):
    #     if config.categorical_features["encoding"] == "onehot":
    #         return 1
    #     elif config.categorical_features["encoding"] == "ordinal":
    #         return normalized_value
    #     elif config.categorical_features["encoding"] == "frequency":
    #         raise NotImplementedError

    def _initialize_onehot_encoder(self, dataset):
        """Initialize the OneHotEncoder based on the categorical values in the dataset."""
        # Extract unique values for the feature from the dataset
        unique_values = dataset.data[self.feature_name].unique().reshape(-1, 1)
        
        # Initialize and fit the OneHotEncoder
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(unique_values)

    def _initialize_label_encoder(self, dataset):
        """Initialize the LabelEncoder based on the categorical values in the dataset."""
        # Extract unique values for the feature from the dataset
        unique_values = dataset.data[self.feature_name].unique().reshape(-1, 1)
        
        # Initialize and fit the LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(unique_values)

    def _onehot_enc(self, value):
        """Encode a categorical value to one-hot vector."""
        # Ensure the encoder is initialized
        if not hasattr(self, 'onehot_encoder'):
            raise ValueError("OneHotEncoder not initialized. Call _initialize_onehot_encoder first.")
            
        # Encode the value
        encoded_value = self.onehot_encoder.transform([[value]])
        return encoded_value[0]  # Return the first (and only) row of the encoded result

    def _onehot_dec(self, encoded_value):
        """Decode a one-hot vector to its original categorical value."""
        # Ensure the encoder is initialized
        if not hasattr(self, 'onehot_encoder'):
            raise ValueError("OneHotEncoder not initialized. Call _initialize_onehot_encoder first.")
        
        # Decode the value
        decoded_value = self.onehot_encoder.inverse_transform([encoded_value])
        return decoded_value[0][0]  # Return the first (and only) value of the decoded result
    
    def _label_enc(self, value):
        transformed_value = self.label_encoder.transform([value])[0]
        print("Label encoder: ", value, " for feature ", self.feature_name, " is transformed into ", transformed_value)
        return transformed_value
    
    def _label_dec(self, value):
        original_value = self.label_encoder.inverse_transform([value])[0]
        return original_value
    
    def _ordinal_enc(self, value):
        return value

    def _freq_enc(self, value):
        raise NotImplementedError

    def _ordinal_dec(self, value):
        return value

    def _freq_dec(self, value):
        raise NotImplementedError
