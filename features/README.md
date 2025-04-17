# Custom Feature Generators

This directory contains custom feature generators that can be used by the trading bot, without modifying the core code in `/src`.

## How to Add a New Feature Generator

1. Create a new Python file in this directory (e.g., `my_feature.py`)
2. Import the base class: `from base import FeatureGeneratorBase`
3. Create a class that inherits from `FeatureGeneratorBase`
4. Implement the `generate(self, df)` method
5. Name your class with the suffix `Generator` (e.g., `MyFeatureGenerator`)
6. Add the feature to the configuration file in `config/model/enhanced_bayesian.yaml`

## Example:

```python
from base import FeatureGeneratorBase
import pandas as pd
import numpy as np

class MyFeatureGenerator(FeatureGeneratorBase):
    """My custom feature generator"""
    
    def generate(self, df):
        # Always make a copy to avoid modifying the original
        df = df.copy()
        
        # Get parameters from configuration
        param1 = self.params.get('model', 'features', 'my_feature', 'param1', default=5)
        
        # Generate features
        df['my_custom_feature'] = df['close'].diff(param1) / df['close']
        
        self.logger.debug(f"Generated my custom features with param1={param1}")
        return df
```
Then in config:
```yaml
model:
  features:
    my_feature:
      enabled: true
      param1: 5
```

## Benefits of This Approach

1. **Separation of Core & Custom Code**: Core feature engineering remains in `/src/`, while custom features live in `/features/`
2. **Easy Integration**: Custom features are automatically discovered and integrated
3. **Simple for Researchers**: Researchers only need to create a simple Python file following a template
4. **Flexible Naming**: Supports multiple naming conventions for generator classes
5. **Fallback Mechanism**: Will try internal features if a custom one isn't found
6. **Sandboxed Development**: Researchers can develop in notebooks and simply copy into the features directory

This approach gives you the best of both worlds - core code integrity and flexibility for researchers to add custom features without touching the main codebase.