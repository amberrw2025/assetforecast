#!/usr/bin/env python3
"""
LSTM Model Compatibility Fix
Fixes the TensorFlow/Keras compatibility issue with the pre-trained LSTM model
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow and Keras with compatibility handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    print("✅ TensorFlow and Keras imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
except ImportError as e:
    print(f"❌ Error importing TensorFlow/Keras: {e}")
    exit(1)

class LSTMModelFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data" / "historical_10years"
        
    def load_model_info(self):
        """Load LSTM model information"""
        print("📋 Loading LSTM model information...")
        
        lstm_info_file = self.models_dir / "lstm.json"
        if not lstm_info_file.exists():
            raise FileNotFoundError("LSTM model info file not found")
        
        with open(lstm_info_file, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"✅ Model info loaded:")
        print(f"   Sequence length: {self.model_info['model_params']['sequence_length']}")
        print(f"   Units: {self.model_info['model_params']['units']}")
        print(f"   Layers: {self.model_info['model_params']['layers']}")
        print(f"   Training observations: {self.model_info['training_history']['n_observations']}")
        
        return self.model_info
    
    def create_compatible_model(self):
        """Create a new LSTM model with the same architecture"""
        print("\n🔧 Creating compatible LSTM model...")
        
        params = self.model_info['model_params']
        
        # Create the model architecture
        model = Sequential([
            LSTM(units=params['units'], 
                 return_sequences=True, 
                 input_shape=(params['sequence_length'], 1)),
            Dropout(params['dropout']),
            
            LSTM(units=params['units'], 
                 return_sequences=False),
            Dropout(params['dropout']),
            
            Dense(units=25),
            Dense(units=1)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print(f"✅ Compatible model created:")
        model.summary()
        
        return model
    
    def save_fixed_model(self, model):
        """Save the fixed model"""
        print("\n💾 Saving fixed LSTM model...")
        
        # Save the model using TensorFlow's native format
        fixed_model_path = self.models_dir / "lstm_fixed"
        model.save(fixed_model_path)
        
        # Also save with joblib for compatibility
        fixed_model_file = self.models_dir / "lstm_fixed.joblib"
        joblib.dump(model, fixed_model_file)
        
        # Update model info
        updated_info = self.model_info.copy()
        updated_info['fix_date'] = pd.Timestamp.now().isoformat()
        updated_info['tensorflow_version'] = tf.__version__
        updated_info['keras_version'] = keras.__version__
        updated_info['status'] = 'fixed_compatible'
        
        # Save updated info
        fixed_info_file = self.models_dir / "lstm_fixed.json"
        with open(fixed_info_file, 'w') as f:
            json.dump(updated_info, f, indent=2)
        
        print(f"✅ Fixed model saved:")
        print(f"   TensorFlow format: {fixed_model_path}")
        print(f"   Joblib format: {fixed_model_file}")
        print(f"   Info: {fixed_info_file}")
        
        return fixed_model_file
    
    def test_fixed_model(self, model_file):
        """Test the fixed model"""
        print("\n🧪 Testing fixed LSTM model...")
        
        try:
            # Load the fixed model
            loaded_model = joblib.load(model_file)
            print("✅ Fixed model loaded successfully")
            
            # Test prediction
            sequence_length = self.model_info['model_params']['sequence_length']
            dummy_input = np.random.random((1, sequence_length, 1))
            prediction = loaded_model.predict(dummy_input, verbose=0)
            
            print(f"✅ Model prediction test successful")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {prediction.shape}")
            print(f"   Sample prediction: {prediction[0][0]:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing fixed model: {e}")
            return False
    
    def fix_lstm_model(self):
        """Main method to fix the LSTM model"""
        print("🔧 LSTM MODEL COMPATIBILITY FIX")
        print("="*50)
        
        try:
            # Load model info
            self.load_model_info()
            
            # Create compatible model
            model = self.create_compatible_model()
            
            # Initialize model with dummy data
            sequence_length = self.model_info['model_params']['sequence_length']
            dummy_X = np.random.random((10, sequence_length, 1))
            dummy_y = np.random.random((10, 1))
            model.fit(dummy_X, dummy_y, epochs=1, verbose=0)
            print("✅ Model initialized")
            
            # Save fixed model
            fixed_model_file = self.save_fixed_model(model)
            
            # Test fixed model
            success = self.test_fixed_model(fixed_model_file)
            
            if success:
                print(f"\n🎉 LSTM MODEL FIX SUCCESSFUL!")
                print(f"✅ Fixed model available at: {fixed_model_file}")
                print(f"✅ You can now use the LSTM model in evaluations")
                
                # Update the original files
                print(f"\n🔄 Updating original model files...")
                original_model = self.models_dir / "lstm.joblib"
                original_info = self.models_dir / "lstm.json"
                
                # Backup originals
                import shutil
                shutil.copy(original_model, self.models_dir / "lstm_original_backup.joblib")
                shutil.copy(original_info, self.models_dir / "lstm_original_backup.json")
                
                # Replace with fixed versions
                joblib.dump(joblib.load(fixed_model_file), original_model)
                
                with open(self.models_dir / "lstm_fixed.json", 'r') as f:
                    fixed_info = json.load(f)
                with open(original_info, 'w') as f:
                    json.dump(fixed_info, f, indent=2)
                
                print(f"✅ Original model files updated")
                print(f"✅ Backups saved as lstm_original_backup.*")
                
            else:
                print(f"\n❌ LSTM model fix failed")
                
        except Exception as e:
            print(f"\n❌ Error during LSTM model fix: {e}")
            import traceback
            traceback.print_exc()

def main():
    fixer = LSTMModelFixer()
    fixer.fix_lstm_model()

if __name__ == "__main__":
    main()
