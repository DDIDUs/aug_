class EXP_MODES:
    ORIGINAL = 1
    DYNAMIC_AUG_ONLY = 2
    ORIG_PLUS_DYNAMIC_AUG_1X = 3    # exclude Augmentation of validation data
    ORIG_PLUS_DYNAMIC_AUG_2X = 4    
    ORIG_PLUS_VALID_AUG_1X = 5      # include Augmentation of validation data
    ORIG_PLUS_VALID_AUG_2X = 6      
