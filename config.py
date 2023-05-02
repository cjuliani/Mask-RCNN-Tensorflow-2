# Directories of training and inference results.
RESULTS_FOLDER = 'results'
SUMMARY_PATH = RESULTS_FOLDER + '/summary'  # folder path of summary results
SAVE_WEIGHTS_PATH = RESULTS_FOLDER + '/weights'  # folder path where to save parameters of training models

# ----- TRAINING CONFIGURATION
INPUT_SIZE = (512, 512)  # input of learning model resized to dimension (width, height)
MODEL_NAME = 'model_001'  # model name define by user
MODEL_FOLDER_TO_RESTORE = 'model_001'

# Training configuration
EPOCHS = 100  # total number of training epochs
SUMMARY_STEP = 1  # step interval for saving summary
VALIDATION_STEP = 5  # step interval for validation while training.
METRIC_RESET_STEP = 1  # reset metric states at every given step
BATCH_SIZE = 2  # /!\ the model only work with batch of 1 (per GPU) /!\

# Learning scheme.
LEARNING_RATE = 1e-4  # if cyclic decay, this corresponds to maximum learning rate
MINIMUM_LEARNING_RATE = 5e-6  # if cyclic decay, this corresponds to base learning rate
MOMENTUM = 0.9  # SGD parameter, only relevant if ADAM_OPTIMIZER is False
NESTEROV = True  # SGD parameter, only relevant if ADAM_OPTIMIZER is False
LR_DECAY = True
LR_DECAY_STEPS = 100  # decay step
LR_DECAY_RATE = 0.90
LR_DECAY_STAIRCASE = True  # if True, use a staircase function type for decay
LR_CYCLIC = False  # if True, apply cyclic LR (/!\ LR_DECAY must also be True to work /!\)
CYCLIC_STEP_SIZE = 50  # number of training iterations per half cycle
CYCLIC_GAMMA = 0.9991  # function amplitude scaling factor
ADAM_OPTIMIZER = True  # if True, use Adam optimization (SGD otherwise)

# Data configuration.
DATA_FOLDER = r"data/iSAID-DOTAv1"
CLASSES_TO_LEARN = ['plane', 'ship']
CATEGORIES_TO_LEARN = [4, 5]
NUM_CLASSES = len(CLASSES_TO_LEARN) + 1  # number of classes + 1 (background)
AUGMENT = True  # data augmentation while training (set to False for validation)

# Saving strategy.
NUM_SAVING_PER_EPOCH = 30  # number of time saving occurs, ie if data_size // number % step is 0

# Summary.
SUM_NUM_BOXES_TO_SHOW = 25  # maximum number of boxes to show in summary
SUM_BOXES_MIN_SCORE = 0.5  # minimum score considered to show a box

# Augmentation.
TRANSLATION_RATE = (0, 9, 3)  # range parameters
DIRECTION = [-1, 1]
ANGLE = (0, 360, 90)  # range parameters
SCALING = [0.9, 1.0, 1.1]

# ----- MODEL CONFIGURATION
# Anchors.
ANCHOR_SCALES = [16, 32, 64]
ANCHOR_RATIOS = [0.5, 1, 1.5, 2]
RPN_BATCH_SIZE = 64

# Increase resolution of backbone output or the density of
# anchors spatially (more grid cells). The number indicates
# how many layers with stride 2 from backbone will see their
# stride changed 1 instead, starting from the output layer.
ANCHOR_DENSITY_LEVEL = 2  # default value is 0 (no density increase); each level increase density by power 2

# Increase number of positives by resampling known positive(s)
# in RPN batch. This is to cope with imbalanced data.
RPN_RESAMPLE_POSITIVES = True
RPN_RESAMPLE_RATE = 0.5  # ratio of batch made of positives

# Label thresholds of foreground/background anchors.
RPN_ANCHOR_FG_THRESH = 0.7  # greater or equal than this value
RPN_ANCHOR_BG_THRESH = 0.3  # less than this value
RPN_FG_RATIO = 0.5  # should be 0.3 to 0.5 (not above 0.5)

# NMS.
MAX_RPN_POST_NMS_TOP_N = 2000  # top scoring boxes to keep after applying NMS to RPN proposals
TEST_MAX_RPN_POST_NMS_TOP_N = 128
NMS_THRESH = 0.7  # if high, more boxes chosen for head targets (and more FPs); not used if NMS_SIGMA > 0
NMS_SIGMA = 0.  # decay scores of boxes around best one instead of pruning them (https://arxiv.org/pdf/1704.04503.pdf)

# Regression targets.
BBOX_NORMALIZE_TARGETS = False
BBOX_NORM_MEAN = (0.0, 0.0, 0.0, 0.0)
BBOX_NORM_STD = (1., 1., 0.1, 0.1)

# ROI pooling.
HEAD_PRE_POOL_SIZE = 7  # pooled resolution = PRE_POOL_SIZE*2

# Head foreground/background thresholds.
HEAD_FG_THRESH = 0.5  # greater or equal than this value
HEAD_BG_THRESH_LO = 0.1  # greater equal than this value
HEAD_BATCH_SIZE = 64

# Increase number of positives by resampling known positive(s)
# in HEAD batch. This is to cope with imbalanced data. Note that
# sampling of positives is always limited by HEAD_FG_RATE (maximum),
# and HEAD_RESAMPLE_POSITIVES forces to resample up to this limit.
HEAD_RESAMPLE_POSITIVES = True
HEAD_FG_RATE = 0.5  # always applied, even if no resampling

# Networks parameters
BACKBONE_MODEL = 'resnet101'
RPN_LOSS_WEIGHTS = (1, 30.)  # (score, regression) losses coefficients
RPN_NEG_POS_WEIGHTS = (1., 1.)  # (background, pos)
HEAD_LOSS_WEIGHTS = (1., 500.)  # (class, regression)
BG_OBJ_WEIGHTS = (2., 1., 1.)  # (background, pos)
MASK_LOSS_WEIGHT = 1.

# Weight loadings.
LOAD_IMAGENET_WEIGHTS = True  # apply ImageNet weights to backbone
LOAD_BACKBONE_WEIGHTS = True  # restore backbone weights from restored model (overwrites ImageNet weights)
LOAD_RPN_WEIGHTS = True
LOAD_HEAD_WEIGHTS = True
LOAD_MASK_WEIGHTS = True
LOAD_CHECKPOINT = True

# Training modes.
TRAIN_BACKBONE = False  # if False, freeze all layers of the backbone network (not trained)
TRAIN_RPN = True  # if False, freeze all layers of the RPN network (not trained)
TRAIN_HEAD = True
TRAIN_HEAD_CLS = True  # if False, do not train the classifier of the HEAD network
TRAIN_HEAD_REG = True
TRAIN_MASK = True  # if False, do not train on mask (only boxes and classes predicted)

# Regularization.
BACKBONE_REG_COEF = 1e-5
RPN_REG_COEF = 1e-5

# Labels.
RPN_LABEL_SMOOTHING = False
HEAD_LABEL_SMOOTHING = False
SMOOTH_FACTOR = 0.1

# Path to backbone weight folder.
# WARNING: Do not change. Used for path definition.
PATH_IMAGENET_WEIGHTS = 'src/modules/networks/weights/'
BACKBONE_PATH = 'src/modules/networks/'
