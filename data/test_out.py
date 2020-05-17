@ six . add_metaclass ( abc . ABCMeta ) 
class Loss ( object ) : 
    def __init__ ( self , 
    reduction = losses_utils . Reduction . SUM_OVER_BATCH_SIZE , 
    name = None ) : 
        self . reduction = reduction 
        self . name = name 
    def __call__ ( self , y_true , y_pred , sample_weight = None ) : 

class TensorBoard ( Callback ) : 
    def __init__ ( self , log_dir = './logs ' , 
    histogram_freq = 0 , 
    batch_size = 32 , 
    write_graph = True , 
    write_grads = False , 
    write_images = False , 
    embeddings_freq = 0 , 
    embeddings_layer_names = None , 
    embeddings_metadata = None , 
    embeddings_data = None , 
    update_freq = 'epoch ' ) : 
        super ( TensorBoard , self ) . __init__ ( ) 
        global tf , projector 
        try : 
            import tensorflow as tf 
            from tensorflow . contrib . tensorboard . plugins import projector 
        except ImportError : 
            raise ImportError ( 'You need the TensorFlow (v1) module installed to ' 
            'use TensorBoard. ' ) 
        if K . backend ( ) != 'tensorflow ' : 
            if histogram_freq != 0 : 
                warnings . warn ( 'You are not using the TensorFlow backend. ' 
                'histogram_freq was set to 0 ' ) 
                histogram_freq = 0 
            if write_graph : 
                warnings . warn ( 'You are not using the TensorFlow backend. ' 
                'write_graph was set to False ' ) 
                write_graph = False 
            if write_images : 
                warnings . warn ( 'You are not using the TensorFlow backend. ' 
                'write_images was set to False ' ) 
                write_images = False 
            if embeddings_freq != 0 : 
                warnings . warn ( 'You are not using the TensorFlow backend. ' 
                'embeddings_freq was set to 0 ' ) 
                embeddings_freq = 0 
        self . log_dir = log_dir 
        self . histogram_freq = histogram_freq 
        self . merged = None 
        self . write_graph = write_graph 
        self . write_grads = write_grads 
        self . write_images = write_images 
        self . embeddings_freq = embeddings_freq 
        self . embeddings_layer_names = embeddings_layer_names 
        self . embeddings_metadata = embeddings_metadata or { } 
        self . batch_size = batch_size 
        self . embeddings_data = embeddings_data 
        if update_freq == 'batch ' : 

def fit_generator ( model , 
generator , 
steps_per_epoch = None , 
epochs = 1 , 
verbose = 1 , 
callbacks = None , 
validation_data = None , 
validation_steps = None , 
validation_freq = 1 , 
class_weight = None , 
max_queue_size = 10 , 
workers = 1 , 
use_multiprocessing = False , 
shuffle = True , 
initial_epoch = 0 ) : 
    epoch = initial_epoch 
    do_validation = bool ( validation_data ) 
    model . _make_train_function ( ) 
    if do_validation : 
        model . _make_test_function ( ) 
    use_sequence_api = is_sequence ( generator ) 
    if not use_sequence_api and use_multiprocessing and workers > 1 : 
        warnings . warn ( 
        UserWarning ( 'Using a generator with `use_multiprocessing=True` ' 
        'and multiple workers may duplicate your data. ' 
        'Please consider using the `keras.utils.Sequence ' 
        'class. ' ) ) 

@ keras_modules_injection 
def VGG16 ( * args , ** kwargs ) : 
    return vgg16 . VGG16 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return vgg16 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return vgg16 . preprocess_input ( * args , ** kwargs ) 

def load_data ( path = 'mnist.npz ' ) : 
    path = get_file ( path , 
    origin = 'https://s3.amazonaws.com/img-datasets/mnist.npz ' , 
    file_hash = '8a61469f7ea1b51cbae51d4f78837e45 ' ) 
    with np . load ( path , allow_pickle = True ) as f : 
        x_train , y_train = f [ 'x_train ' ] , f [ 'y_train ' ] 
        x_test , y_test = f [ 'x_test ' ] , f [ 'y_test ' ] 
    return ( x_train , y_train ) , ( x_test , y_test ) 


NEG_INF = - 1e10 
class Reduction ( object ) : 
    SUM = 'sum ' 
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size ' 
    WEIGHTED_MEAN = 'weighted_mean ' 
def update_state_wrapper ( update_state_fn ) : 
    def decorated ( metric_obj , * args , ** kwargs ) : 
        update_op = update_state_fn ( * args , ** kwargs ) 
        metric_obj . add_update ( update_op ) 
        return update_op 
    return decorated 
def result_wrapper ( result_fn ) : 
    def decorated ( metric_obj , * args , ** kwargs ) : 
        result_t = K . identity ( result_fn ( * args , ** kwargs ) ) 
        metric_obj . _call_result = result_t 
        result_t . _is_metric = True 
        return result_t 
    return decorated 
def filter_top_k ( x , k ) : 
    import tensorflow as tf 
    _ , top_k_idx = tf . nn . top_k ( x , k , sorted = False ) 
    top_k_mask = K . sum ( 
    K . one_hot ( top_k_idx , x . shape [ - 1 ] ) , axis = - 2 ) 
    return x * top_k_mask + NEG_INF * ( 1 - top_k_mask ) 
def to_list ( x ) : 
    if isinstance ( x , list ) : 
        return x 
    return [ x ] 
def assert_thresholds_range ( thresholds ) : 
    if thresholds is not None : 
        invalid_thresholds = [ t for t in thresholds if t is None or t < 0 or t > 1 ] 
    if invalid_thresholds : 
        raise ValueError ( 
        'Threshold values must be in [0, 1]. Invalid values: {} ' . format ( 
        invalid_thresholds ) ) 
def parse_init_thresholds ( thresholds , default_threshold = 0.5 ) : 
    if thresholds is not None : 
        assert_thresholds_range ( to_list ( thresholds ) ) 
    thresholds = to_list ( default_threshold if thresholds is None else thresholds ) 
    return thresholds 
class ConfusionMatrix ( Enum ) : 
    TRUE_POSITIVES = 'tp ' 
    FALSE_POSITIVES = 'fp ' 
    TRUE_NEGATIVES = 'tn ' 
    FALSE_NEGATIVES = 'fn ' 
class AUCCurve ( Enum ) : 
    ROC = 'ROC ' 
    PR = 'PR ' 
    @ staticmethod 
    def from_str ( key ) : 
        if key in ( 'pr ' , 'PR ' ) : 
            return AUCCurve . PR 
        elif key in ( 'roc ' , 'ROC ' ) : 
            return AUCCurve . ROC 
        else : 
            raise ValueError ( 'Invalid AUC curve value "%s". ' % key ) 
class AUCSummationMethod ( Enum ) : 
    INTERPOLATION = 'interpolation ' 
    MAJORING = 'majoring ' 
    MINORING = 'minoring ' 
    @ staticmethod 
    def from_str ( key ) : 
        if key in ( 'interpolation ' , 'Interpolation ' ) : 
            return AUCSummationMethod . INTERPOLATION 
        elif key in ( 'majoring ' , 'Majoring ' ) : 
            return AUCSummationMethod . MAJORING 
        elif key in ( 'minoring ' , 'Minoring ' ) : 
            return AUCSummationMethod . MINORING 
        else : 
            raise ValueError ( 'Invalid AUC summation method value "%s". ' % key ) 
def weighted_assign_add ( label , pred , weights , var ) : 


class Embedding ( Layer ) : 
    @ interfaces . legacy_embedding_support 
    def __init__ ( self , input_dim , output_dim , 
    embeddings_initializer = 'uniform ' , 
    embeddings_regularizer = None , 
    activity_regularizer = None , 
    embeddings_constraint = None , 
    mask_zero = False , 
    input_length = None , 
    ** kwargs ) : 
        if 'input_shape ' not in kwargs : 
            if input_length : 
                kwargs [ 'input_shape ' ] = ( input_length , ) 
            else : 
                kwargs [ 'input_shape ' ] = ( None , ) 
        super ( Embedding , self ) . __init__ ( ** kwargs ) 
        self . input_dim = input_dim 
        self . output_dim = output_dim 
        self . embeddings_initializer = initializers . get ( embeddings_initializer ) 
        self . embeddings_regularizer = regularizers . get ( embeddings_regularizer ) 
        self . activity_regularizer = regularizers . get ( activity_regularizer ) 
        self . embeddings_constraint = constraints . get ( embeddings_constraint ) 
        self . mask_zero = mask_zero 
        self . supports_masking = mask_zero 
        self . input_length = input_length 
    def build ( self , input_shape ) : 
        self . embeddings = self . add_weight ( 
        shape = ( self . input_dim , self . output_dim ) , 
        initializer = self . embeddings_initializer , 
        name = 'embeddings ' , 
        regularizer = self . embeddings_regularizer , 
        constraint = self . embeddings_constraint , 
        dtype = self . dtype ) 
        self . built = True 
    def compute_mask ( self , inputs , mask = None ) : 
        if not self . mask_zero : 
            return None 
        output_mask = K . not_equal ( inputs , 0 ) 
        return output_mask 
    def compute_output_shape ( self , input_shape ) : 
        if self . input_length is None : 
            return input_shape + ( self . output_dim , ) 
        else : 

try : 
    import queue 
except ImportError : 
    import Queue as queue 
if sys . version_info [ 0 ] == 2 : 
    def urlretrieve ( url , filename , reporthook = None , data = None ) : 
        def chunk_read ( response , chunk_size = 8192 , reporthook = None ) : 
            content_type = response . info ( ) . get ( 'Content-Length ' ) 
            total_size = - 1 
            if content_type is not None : 
                total_size = int ( content_type . strip ( ) ) 
            count = 0 
            while True : 
                chunk = response . read ( chunk_size ) 
                count += 1 
                if reporthook is not None : 
                    reporthook ( count , chunk_size , total_size ) 
                if chunk : 
                    yield chunk 
                else : 
                    break 
        with closing ( urlopen ( url , data ) ) as response , open ( filename , 'wb ' ) as fd : 
            for chunk in chunk_read ( response , reporthook = reporthook ) : 
                fd . write ( chunk ) 
else : 
    from six . moves . urllib . request import urlretrieve 
def _extract_archive ( file_path , path = '. ' , archive_format = 'auto ' ) : 
    if archive_format is None : 
        return False 
    if archive_format == 'auto ' : 
        archive_format = [ 'tar ' , 'zip ' ] 
    if isinstance ( archive_format , six . string_types ) : 
        archive_format = [ archive_format ] 
    for archive_type in archive_format : 
        if archive_type == 'tar ' : 
            open_fn = tarfile . open 
            is_match_fn = tarfile . is_tarfile 
        if archive_type == 'zip ' : 
            open_fn = zipfile . ZipFile 
            is_match_fn = zipfile . is_zipfile 
        if is_match_fn ( file_path ) : 
            with open_fn ( file_path ) as archive : 
                try : 
                    archive . extractall ( path ) 
                except ( tarfile . TarError , RuntimeError , 
                KeyboardInterrupt ) : 
                    if os . path . exists ( path ) : 
                        if os . path . isfile ( path ) : 
                            os . remove ( path ) 
                        else : 
                            shutil . rmtree ( path ) 
                    raise 
            return True 
    return False 
def get_file ( fname , 
origin , 
untar = False , 
md5_hash = None , 
file_hash = None , 
cache_subdir = 'datasets ' , 
hash_algorithm = 'auto ' , 
extract = False , 
archive_format = 'auto ' , 
cache_dir = None ) : 

py_all = all 
py_any = any 
py_sum = sum 
py_slice = slice 

class Constraint ( object ) : 
    def __call__ ( self , w ) : 
        return w 
    def get_config ( self ) : 
        return { } 
class MaxNorm ( Constraint ) : 
    def __init__ ( self , max_value = 2 , axis = 0 ) : 
        self . max_value = max_value 
        self . axis = axis 
    def __call__ ( self , w ) : 
        norms = K . sqrt ( K . sum ( K . square ( w ) , axis = self . axis , keepdims = True ) ) 
        desired = K . clip ( norms , 0 , self . max_value ) 
        return w * ( desired / ( K . epsilon ( ) + norms ) ) 
    def get_config ( self ) : 
        return { 'max_value ' : self . max_value , 
        'axis ' : self . axis } 
class NonNeg ( Constraint ) : 
    def __call__ ( self , w ) : 
        return w * K . cast ( K . greater_equal ( w , 0. ) , K . floatx ( ) ) 
class UnitNorm ( Constraint ) : 
    def __init__ ( self , axis = 0 ) : 
        self . axis = axis 
    def __call__ ( self , w ) : 
        return w / ( K . epsilon ( ) + K . sqrt ( K . sum ( K . square ( w ) , 
        axis = self . axis , 
        keepdims = True ) ) ) 
    def get_config ( self ) : 
        return { 'axis ' : self . axis } 
class MinMaxNorm ( Constraint ) : 
    def __init__ ( self , min_value = 0.0 , max_value = 1.0 , rate = 1.0 , axis = 0 ) : 
        self . min_value = min_value 
        self . max_value = max_value 
        self . rate = rate 
        self . axis = axis 
    def __call__ ( self , w ) : 
        norms = K . sqrt ( K . sum ( K . square ( w ) , axis = self . axis , keepdims = True ) ) 
        desired = ( self . rate * K . clip ( norms , self . min_value , self . max_value ) + 
        ( 1 - self . rate ) * norms ) 
        return w * ( desired / ( K . epsilon ( ) + norms ) ) 
    def get_config ( self ) : 
        return { 'min_value ' : self . min_value , 
        'max_value ' : self . max_value , 
        'rate ' : self . rate , 
        'axis ' : self . axis } 

try : 
    import requests 
except ImportError : 
    requests = None 
_TRAIN = 'train ' 
_TEST = 'test ' 
_PREDICT = 'predict ' 
class CallbackList ( object ) : 
    def __init__ ( self , callbacks = None , queue_length = 10 ) : 
        callbacks = callbacks or [ ] 
        self . callbacks = [ c for c in callbacks ] 
        self . queue_length = queue_length 
        self . params = { } 
        self . model = None 
        self . _reset_batch_timing ( ) 
    def _reset_batch_timing ( self ) : 
        self . _delta_t_batch = 0. 
        self . _delta_ts = defaultdict ( lambda : deque ( [ ] , maxlen = self . queue_length ) ) 
    def append ( self , callback ) : 
        self . callbacks . append ( callback ) 
    def set_params ( self , params ) : 
        self . params = params 
        for callback in self . callbacks : 
            callback . set_params ( params ) 
    def set_model ( self , model ) : 
        self . model = model 
        for callback in self . callbacks : 
            callback . set_model ( model ) 
    def _call_batch_hook ( self , mode , hook , batch , logs = None ) : 
        if not self . callbacks : 
            return 
        hook_name = 'on_{mode}_batch_{hook} ' . format ( mode = mode , hook = hook ) 
        if hook == 'end ' : 
            if not hasattr ( self , '_t_enter_batch ' ) : 
                self . _t_enter_batch = time . time ( ) 

random_rotation = image . random_rotation 
random_shift = image . random_shift 
random_shear = image . random_shear 
random_zoom = image . random_zoom 
apply_channel_shift = image . apply_channel_shift 
random_channel_shift = image . random_channel_shift 
apply_brightness_shift = image . apply_brightness_shift 
random_brightness = image . random_brightness 
apply_affine_transform = image . apply_affine_transform 
load_img = image . load_img 
def array_to_img ( x , data_format = None , scale = True , dtype = None ) : 
    if data_format is None : 
        data_format = backend . image_data_format ( ) 
    if dtype is None : 
        dtype = backend . floatx ( ) 
    return image . array_to_img ( x , 
    data_format = data_format , 
    scale = scale , 
    dtype = dtype ) 
def img_to_array ( img , data_format = None , dtype = None ) : 
    if data_format is None : 
        data_format = backend . image_data_format ( ) 
    if dtype is None : 
        dtype = backend . floatx ( ) 
    return image . img_to_array ( img , data_format = data_format , dtype = dtype ) 
def save_img ( path , 
x , 
data_format = None , 
file_format = None , 
scale = True , ** kwargs ) : 
    if data_format is None : 
        data_format = backend . image_data_format ( ) 
    return image . save_img ( path , 
    x , 
    data_format = data_format , 
    file_format = file_format , 
    scale = scale , ** kwargs ) 
class Iterator ( image . Iterator , utils . Sequence ) : 
    pass 
class DirectoryIterator ( image . DirectoryIterator , Iterator ) : 
    __doc__ = image . DirectoryIterator . __doc__ 
    def __init__ ( self , directory , image_data_generator , 
    target_size = ( 256 , 256 ) , 
    color_mode = 'rgb ' , 
    classes = None , 
    class_mode = 'categorical ' , 
    batch_size = 32 , 
    shuffle = True , 
    seed = None , 
    data_format = None , 
    save_to_dir = None , 
    save_prefix = '' , 
    save_format = 'png ' , 
    follow_links = False , 
    subset = None , 
    interpolation = 'nearest ' , 
    dtype = None ) : 
        if data_format is None : 
            data_format = backend . image_data_format ( ) 
        if dtype is None : 
            dtype = backend . floatx ( ) 
        super ( DirectoryIterator , self ) . __init__ ( 
        directory , image_data_generator , 
        target_size = target_size , 
        color_mode = color_mode , 
        classes = classes , 
        class_mode = class_mode , 
        batch_size = batch_size , 
        shuffle = shuffle , 
        seed = seed , 
        data_format = data_format , 
        save_to_dir = save_to_dir , 
        save_prefix = save_prefix , 
        save_format = save_format , 
        follow_links = follow_links , 
        subset = subset , 
        interpolation = interpolation , 
        dtype = dtype ) 
class NumpyArrayIterator ( image . NumpyArrayIterator , Iterator ) : 
    __doc__ = image . NumpyArrayIterator . __doc__ 
    def __init__ ( self , x , y , image_data_generator , 
    batch_size = 32 , 
    shuffle = False , 
    sample_weight = None , 
    seed = None , 
    data_format = None , 
    save_to_dir = None , 
    save_prefix = '' , 
    save_format = 'png ' , 
    subset = None , 
    dtype = None ) : 
        if data_format is None : 
            data_format = backend . image_data_format ( ) 
        if dtype is None : 
            dtype = backend . floatx ( ) 
        super ( NumpyArrayIterator , self ) . __init__ ( 
        x , y , image_data_generator , 
        batch_size = batch_size , 
        shuffle = shuffle , 
        sample_weight = sample_weight , 
        seed = seed , 
        data_format = data_format , 
        save_to_dir = save_to_dir , 
        save_prefix = save_prefix , 
        save_format = save_format , 
        subset = subset , 
        dtype = dtype ) 
class DataFrameIterator ( image . DataFrameIterator , Iterator ) : 
    __doc__ = image . DataFrameIterator . __doc__ 
    def __init__ ( self , 
    dataframe , 
    directory = None , 
    image_data_generator = None , 
    x_col = 'filename ' , 
    y_col = 'class ' , 
    weight_col = None , 
    target_size = ( 256 , 256 ) , 
    color_mode = 'rgb ' , 
    classes = None , 
    class_mode = 'categorical ' , 
    batch_size = 32 , 
    shuffle = True , 
    seed = None , 
    data_format = 'channels_last ' , 
    save_to_dir = None , 
    save_prefix = '' , 
    save_format = 'png ' , 
    subset = None , 
    interpolation = 'nearest ' , 
    dtype = 'float32 ' , 
    validate_filenames = True ) : 
        if data_format is None : 
            data_format = backend . image_data_format ( ) 
        if dtype is None : 
            dtype = backend . floatx ( ) 
        super ( DataFrameIterator , self ) . __init__ ( 
        dataframe , 
        directory = directory , 
        image_data_generator = image_data_generator , 
        x_col = x_col , 
        y_col = y_col , 
        weight_col = weight_col , 
        target_size = target_size , 
        color_mode = color_mode , 
        classes = classes , 
        class_mode = class_mode , 
        batch_size = batch_size , 
        shuffle = shuffle , 
        seed = seed , 
        data_format = data_format , 
        save_to_dir = save_to_dir , 
        save_prefix = save_prefix , 
        save_format = save_format , 
        subset = subset , 
        interpolation = interpolation , 
        dtype = dtype , 
        validate_filenames = validate_filenames ) 
class ImageDataGenerator ( image . ImageDataGenerator ) : 
    __doc__ = image . ImageDataGenerator . __doc__ 
    def __init__ ( self , 
    featurewise_center = False , 
    samplewise_center = False , 
    featurewise_std_normalization = False , 
    samplewise_std_normalization = False , 
    zca_whitening = False , 
    zca_epsilon = 1e-6 , 
    rotation_range = 0 , 
    width_shift_range = 0. , 
    height_shift_range = 0. , 
    brightness_range = None , 
    shear_range = 0. , 
    zoom_range = 0. , 
    channel_shift_range = 0. , 
    fill_mode = 'nearest ' , 
    cval = 0. , 
    horizontal_flip = False , 
    vertical_flip = False , 
    rescale = None , 
    preprocessing_function = None , 
    data_format = 'channels_last ' , 
    validation_split = 0.0 , 
    interpolation_order = 1 , 
    dtype = 'float32 ' ) : 
        if data_format is None : 
            data_format = backend . image_data_format ( ) 
        if dtype is None : 
            dtype = backend . floatx ( ) 
        super ( ImageDataGenerator , self ) . __init__ ( 
        featurewise_center = featurewise_center , 
        samplewise_center = samplewise_center , 
        featurewise_std_normalization = featurewise_std_normalization , 
        samplewise_std_normalization = samplewise_std_normalization , 
        zca_whitening = zca_whitening , 
        zca_epsilon = zca_epsilon , 
        rotation_range = rotation_range , 
        width_shift_range = width_shift_range , 
        height_shift_range = height_shift_range , 
        brightness_range = brightness_range , 
        shear_range = shear_range , 
        zoom_range = zoom_range , 
        channel_shift_range = channel_shift_range , 
        fill_mode = fill_mode , 
        cval = cval , 
        horizontal_flip = horizontal_flip , 
        vertical_flip = vertical_flip , 
        rescale = rescale , 
        preprocessing_function = preprocessing_function , 
        data_format = data_format , 
        validation_split = validation_split , 
        interpolation_order = interpolation_order , 
        dtype = dtype ) 
    def flow ( self , 
    x , 
    y = None , 
    batch_size = 32 , 
    shuffle = True , 
    sample_weight = None , 
    seed = None , 
    save_to_dir = None , 
    save_prefix = '' , 
    save_format = 'png ' , 
    subset = None ) : 
        return NumpyArrayIterator ( 
        x , 
        y , 
        self , 
        batch_size = batch_size , 
        shuffle = shuffle , 
        sample_weight = sample_weight , 
        seed = seed , 
        data_format = self . data_format , 
        save_to_dir = save_to_dir , 
        save_prefix = save_prefix , 
        save_format = save_format , 
        subset = subset 
        ) 
    def flow_from_directory ( self , 
    directory , 
    target_size = ( 256 , 256 ) , 
    color_mode = 'rgb ' , 
    classes = None , 
    class_mode = 'categorical ' , 
    batch_size = 32 , 
    shuffle = True , 
    seed = None , 
    save_to_dir = None , 
    save_prefix = '' , 
    save_format = 'png ' , 
    follow_links = False , 
    subset = None , 
    interpolation = 'nearest ' ) : 
        return DirectoryIterator ( 
        directory , 
        self , 
        target_size = target_size , 
        color_mode = color_mode , 
        classes = classes , 
        class_mode = class_mode , 
        data_format = self . data_format , 
        batch_size = batch_size , 
        shuffle = shuffle , 
        seed = seed , 
        save_to_dir = save_to_dir , 
        save_prefix = save_prefix , 
        save_format = save_format , 
        follow_links = follow_links , 
        subset = subset , 
        interpolation = interpolation 
        ) 
    def flow_from_dataframe ( self , 
    dataframe , 
    directory = None , 
    x_col = 'filename ' , 
    y_col = 'class ' , 
    weight_col = None , 
    target_size = ( 256 , 256 ) , 
    color_mode = 'rgb ' , 
    classes = None , 
    class_mode = 'categorical ' , 
    batch_size = 32 , 
    shuffle = True , 
    seed = None , 
    save_to_dir = None , 
    save_prefix = '' , 
    save_format = 'png ' , 
    subset = None , 
    interpolation = 'nearest ' , 
    validate_filenames = True , 
    ** kwargs ) : 
        return DataFrameIterator ( 
        dataframe , 
        directory , 
        self , 
        x_col = x_col , 
        y_col = y_col , 
        weight_col = weight_col , 
        target_size = target_size , 
        color_mode = color_mode , 
        classes = classes , 
        class_mode = class_mode , 
        data_format = self . data_format , 
        batch_size = batch_size , 
        shuffle = shuffle , 
        seed = seed , 
        save_to_dir = save_to_dir , 
        save_prefix = save_prefix , 
        save_format = save_format , 
        subset = subset , 
        interpolation = interpolation , 
        validate_filenames = validate_filenames , 
        ** kwargs 
        ) 
array_to_img . __doc__ = image . array_to_img . __doc__ 
img_to_array . __doc__ = image . img_to_array . __doc__ 
save_img . __doc__ = image . save_img . __doc__ 

try : 
    from keras_applications import resnet 
except : 
    resnet = None 
@ keras_modules_injection 
def ResNet50 ( * args , ** kwargs ) : 
    return resnet . ResNet50 ( * args , ** kwargs ) 
@ keras_modules_injection 
def ResNet101 ( * args , ** kwargs ) : 
    return resnet . ResNet101 ( * args , ** kwargs ) 
@ keras_modules_injection 
def ResNet152 ( * args , ** kwargs ) : 
    return resnet . ResNet152 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return resnet . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return resnet . preprocess_input ( * args , ** kwargs ) 

def generate_legacy_interface ( allowed_positional_args = None , 
conversions = None , 
preprocessor = None , 
value_conversions = None , 
object_type = 'class ' ) : 
    if allowed_positional_args is None : 
        check_positional_args = False 
    else : 
        check_positional_args = True 
    allowed_positional_args = allowed_positional_args or [ ] 
    conversions = conversions or [ ] 
    value_conversions = value_conversions or [ ] 
    def legacy_support ( func ) : 
        @ six . wraps ( func ) 
        def wrapper ( * args , ** kwargs ) : 
            if object_type == 'class ' : 
                object_name = args [ 0 ] . __class__ . __name__ 
            else : 
                object_name = func . __name__ 
            if preprocessor : 
                args , kwargs , converted = preprocessor ( args , kwargs ) 
            else : 
                converted = [ ] 
            if check_positional_args : 
                if len ( args ) > len ( allowed_positional_args ) + 1 : 
                    raise TypeError ( '` ' + object_name + 
                    '` can accept only ' + 
                    str ( len ( allowed_positional_args ) ) + 
                    'positional arguments ' + 
                    str ( tuple ( allowed_positional_args ) ) + 
                    ', but you passed the following ' 
                    'positional arguments: ' + 
                    str ( list ( args [ 1 : ] ) ) ) 
            for key in value_conversions : 
                if key in kwargs : 
                    old_value = kwargs [ key ] 
                    if old_value in value_conversions [ key ] : 
                        kwargs [ key ] = value_conversions [ key ] [ old_value ] 
            for old_name , new_name in conversions : 
                if old_name in kwargs : 
                    value = kwargs . pop ( old_name ) 
                    if new_name in kwargs : 
                        raise_duplicate_arg_error ( old_name , new_name ) 
                    kwargs [ new_name ] = value 
                    converted . append ( ( new_name , old_name ) ) 
            if converted : 
                signature = '` ' + object_name + '( ' 
                for i , value in enumerate ( args [ 1 : ] ) : 
                    if isinstance ( value , six . string_types ) : 
                        signature += '" '+ value + ' '' 
                    else : 
                        if isinstance ( value , np . ndarray ) : 
                            str_val = 'array ' 
                        else : 
                            str_val = str ( value ) 
                        if len ( str_val ) > 10 : 
                            str_val = str_val [ : 10 ] + '... ' 
                        signature += str_val 
                    if i < len ( args [ 1 : ] ) - 1 or kwargs : 
                        signature += ', ' 
                for i , ( name , value ) in enumerate ( kwargs . items ( ) ) : 
                    signature += name + '= ' 
                    if isinstance ( value , six . string_types ) : 
                        signature += '" '+ value + ' '' 
                    else : 
                        if isinstance ( value , np . ndarray ) : 
                            str_val = 'array ' 
                        else : 
                            str_val = str ( value ) 
                        if len ( str_val ) > 10 : 
                            str_val = str_val [ : 10 ] + '... ' 
                        signature += str_val 
                    if i < len ( kwargs ) - 1 : 
                        signature += ', ' 
                signature += ')` ' 
                warnings . warn ( 'Update your ` ' + object_name + '` call to the ' + 
                'Keras 2 API: ' + signature , stacklevel = 2 ) 
            return func ( * args , ** kwargs ) 
        wrapper . _original_function = func 
        return wrapper 
    return legacy_support 
generate_legacy_method_interface = functools . partial ( generate_legacy_interface , 
object_type = 'method ' ) 
def raise_duplicate_arg_error ( old_arg , new_arg ) : 
    raise TypeError ( 'For the ` ' + new_arg + '` argument, ' 
    'the layer received both ' 
    'the legacy keyword argument ' 
    '` ' + old_arg + '` and the Keras 2 keyword argument ' 
    '` ' + new_arg + '`. Stick to the latter! ' ) 
legacy_dense_support = generate_legacy_interface ( 
allowed_positional_args = [ 'units ' ] , 
conversions = [ ( 'output_dim ' , 'units ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'W_constraint ' , 'kernel_constraint ' ) , 
( 'b_constraint ' , 'bias_constraint ' ) , 
( 'bias ' , 'use_bias ' ) ] ) 
legacy_dropout_support = generate_legacy_interface ( 
allowed_positional_args = [ 'rate ' , 'noise_shape ' , 'seed ' ] , 
conversions = [ ( 'p ' , 'rate ' ) ] ) 
def embedding_kwargs_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if 'dropout ' in kwargs : 
        kwargs . pop ( 'dropout ' ) 
        warnings . warn ( 'The `dropout` argument is no longer support in `Embedding`. ' 
        'You can apply a `keras.layers.SpatialDropout1D` layer ' 
        'right after the `Embedding` layer to get the same behavior. ' , 
        stacklevel = 3 ) 
    return args , kwargs , converted 
legacy_embedding_support = generate_legacy_interface ( 
allowed_positional_args = [ 'input_dim ' , 'output_dim ' ] , 
conversions = [ ( 'init ' , 'embeddings_initializer ' ) , 
( 'W_regularizer ' , 'embeddings_regularizer ' ) , 
( 'W_constraint ' , 'embeddings_constraint ' ) ] , 
preprocessor = embedding_kwargs_preprocessor ) 
legacy_pooling1d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'pool_size ' , 'strides ' , 'padding ' ] , 
conversions = [ ( 'pool_length ' , 'pool_size ' ) , 
( 'stride ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) ] ) 
legacy_prelu_support = generate_legacy_interface ( 
allowed_positional_args = [ 'alpha_initializer ' ] , 
conversions = [ ( 'init ' , 'alpha_initializer ' ) ] ) 
legacy_gaussiannoise_support = generate_legacy_interface ( 
allowed_positional_args = [ 'stddev ' ] , 
conversions = [ ( 'sigma ' , 'stddev ' ) ] ) 
def recurrent_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if 'forget_bias_init ' in kwargs : 
        if kwargs [ 'forget_bias_init ' ] == 'one ' : 
            kwargs . pop ( 'forget_bias_init ' ) 
            kwargs [ 'unit_forget_bias ' ] = True 
            converted . append ( ( 'forget_bias_init ' , 'unit_forget_bias ' ) ) 
        else : 
            kwargs . pop ( 'forget_bias_init ' ) 
            warnings . warn ( 'The `forget_bias_init` argument ' 
            'has been ignored. Use `unit_forget_bias=True` ' 
            'instead to initialize with ones. ' , stacklevel = 3 ) 
    if 'input_dim ' in kwargs : 
        input_length = kwargs . pop ( 'input_length ' , None ) 
        input_dim = kwargs . pop ( 'input_dim ' ) 
        input_shape = ( input_length , input_dim ) 
        kwargs [ 'input_shape ' ] = input_shape 
        converted . append ( ( 'input_dim ' , 'input_shape ' ) ) 
        warnings . warn ( 'The `input_dim` and `input_length` arguments ' 
        'in recurrent layers are deprecated. ' 
        'Use `input_shape` instead. ' , stacklevel = 3 ) 
    return args , kwargs , converted 
legacy_recurrent_support = generate_legacy_interface ( 
allowed_positional_args = [ 'units ' ] , 
conversions = [ ( 'output_dim ' , 'units ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'inner_init ' , 'recurrent_initializer ' ) , 
( 'inner_activation ' , 'recurrent_activation ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'U_regularizer ' , 'recurrent_regularizer ' ) , 
( 'dropout_W ' , 'dropout ' ) , 
( 'dropout_U ' , 'recurrent_dropout ' ) , 
( 'consume_less ' , 'implementation ' ) ] , 
value_conversions = { 'consume_less ' : { 'cpu ' : 0 , 
'mem ' : 1 , 
'gpu ' : 2 } } , 
preprocessor = recurrent_args_preprocessor ) 
legacy_gaussiandropout_support = generate_legacy_interface ( 
allowed_positional_args = [ 'rate ' ] , 
conversions = [ ( 'p ' , 'rate ' ) ] ) 
legacy_pooling2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'pool_size ' , 'strides ' , 'padding ' ] , 
conversions = [ ( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_pooling3d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'pool_size ' , 'strides ' , 'padding ' ] , 
conversions = [ ( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_global_pooling_support = generate_legacy_interface ( 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_upsampling1d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'size ' ] , 
conversions = [ ( 'length ' , 'size ' ) ] ) 
legacy_upsampling2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'size ' ] , 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_upsampling3d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'size ' ] , 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
def conv1d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if 'input_dim ' in kwargs : 
        if 'input_length ' in kwargs : 
            length = kwargs . pop ( 'input_length ' ) 
        else : 
            length = None 
        input_shape = ( length , kwargs . pop ( 'input_dim ' ) ) 
        kwargs [ 'input_shape ' ] = input_shape 
        converted . append ( ( 'input_shape ' , 'input_dim ' ) ) 
    return args , kwargs , converted 
legacy_conv1d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'filters ' , 'kernel_size ' ] , 
conversions = [ ( 'nb_filter ' , 'filters ' ) , 
( 'filter_length ' , 'kernel_size ' ) , 
( 'subsample_length ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'W_constraint ' , 'kernel_constraint ' ) , 
( 'b_constraint ' , 'bias_constraint ' ) , 
( 'bias ' , 'use_bias ' ) ] , 
preprocessor = conv1d_args_preprocessor ) 
def conv2d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if len ( args ) > 4 : 
        raise TypeError ( 'Layer can receive at most 3 positional arguments. ' ) 
    elif len ( args ) == 4 : 
        if isinstance ( args [ 2 ] , int ) and isinstance ( args [ 3 ] , int ) : 
            new_keywords = [ 'padding ' , 'strides ' , 'data_format ' ] 
            for kwd in new_keywords : 
                if kwd in kwargs : 
                    raise ValueError ( 
                    'It seems that you are using the Keras 2 ' 
                    'and you are passing both `kernel_size` and `strides` ' 
                    'as integer positional arguments. For safety reasons, ' 
                    'this is disallowed. Pass `strides` ' 
                    'as a keyword argument instead. ' ) 
            kernel_size = ( args [ 2 ] , args [ 3 ] ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'nb_row/nb_col ' ) ) 
    elif len ( args ) == 3 and isinstance ( args [ 2 ] , int ) : 
        if 'nb_col ' in kwargs : 
            kernel_size = ( args [ 2 ] , kwargs . pop ( 'nb_col ' ) ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'nb_row/nb_col ' ) ) 
    elif len ( args ) == 2 : 
        if 'nb_row ' in kwargs and 'nb_col ' in kwargs : 
            kernel_size = ( kwargs . pop ( 'nb_row ' ) , kwargs . pop ( 'nb_col ' ) ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'nb_row/nb_col ' ) ) 
    elif len ( args ) == 1 : 
        if 'nb_row ' in kwargs and 'nb_col ' in kwargs : 
            kernel_size = ( kwargs . pop ( 'nb_row ' ) , kwargs . pop ( 'nb_col ' ) ) 
            kwargs [ 'kernel_size ' ] = kernel_size 
            converted . append ( ( 'kernel_size ' , 'nb_row/nb_col ' ) ) 
    return args , kwargs , converted 
legacy_conv2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'filters ' , 'kernel_size ' ] , 
conversions = [ ( 'nb_filter ' , 'filters ' ) , 
( 'subsample ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'W_constraint ' , 'kernel_constraint ' ) , 
( 'b_constraint ' , 'bias_constraint ' ) , 
( 'bias ' , 'use_bias ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } , 
preprocessor = conv2d_args_preprocessor ) 
def separable_conv2d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if 'init ' in kwargs : 
        init = kwargs . pop ( 'init ' ) 
        kwargs [ 'depthwise_initializer ' ] = init 
        kwargs [ 'pointwise_initializer ' ] = init 
        converted . append ( ( 'init ' , 'depthwise_initializer/pointwise_initializer ' ) ) 
    args , kwargs , _converted = conv2d_args_preprocessor ( args , kwargs ) 
    return args , kwargs , converted + _converted 
legacy_separable_conv2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'filters ' , 'kernel_size ' ] , 
conversions = [ ( 'nb_filter ' , 'filters ' ) , 
( 'subsample ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'b_constraint ' , 'bias_constraint ' ) , 
( 'bias ' , 'use_bias ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } , 
preprocessor = separable_conv2d_args_preprocessor ) 
def deconv2d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if len ( args ) == 5 : 
        if isinstance ( args [ 4 ] , tuple ) : 
            args = args [ : - 1 ] 
            converted . append ( ( 'output_shape ' , None ) ) 
    if 'output_shape ' in kwargs : 
        kwargs . pop ( 'output_shape ' ) 
        converted . append ( ( 'output_shape ' , None ) ) 
    args , kwargs , _converted = conv2d_args_preprocessor ( args , kwargs ) 
    return args , kwargs , converted + _converted 
legacy_deconv2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'filters ' , 'kernel_size ' ] , 
conversions = [ ( 'nb_filter ' , 'filters ' ) , 
( 'subsample ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'W_constraint ' , 'kernel_constraint ' ) , 
( 'b_constraint ' , 'bias_constraint ' ) , 
( 'bias ' , 'use_bias ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } , 
preprocessor = deconv2d_args_preprocessor ) 
def conv3d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if len ( args ) > 5 : 
        raise TypeError ( 'Layer can receive at most 4 positional arguments. ' ) 
    if len ( args ) == 5 : 
        if all ( [ isinstance ( x , int ) for x in args [ 2 : 5 ] ] ) : 
            kernel_size = ( args [ 2 ] , args [ 3 ] , args [ 4 ] ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'kernel_dim* ' ) ) 
    elif len ( args ) == 4 and isinstance ( args [ 3 ] , int ) : 
        if isinstance ( args [ 2 ] , int ) and isinstance ( args [ 3 ] , int ) : 
            new_keywords = [ 'padding ' , 'strides ' , 'data_format ' ] 
            for kwd in new_keywords : 
                if kwd in kwargs : 
                    raise ValueError ( 
                    'It seems that you are using the Keras 2 ' 
                    'and you are passing both `kernel_size` and `strides` ' 
                    'as integer positional arguments. For safety reasons, ' 
                    'this is disallowed. Pass `strides` ' 
                    'as a keyword argument instead. ' ) 
        if 'kernel_dim3 ' in kwargs : 
            kernel_size = ( args [ 2 ] , args [ 3 ] , kwargs . pop ( 'kernel_dim3 ' ) ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'kernel_dim* ' ) ) 
    elif len ( args ) == 3 : 
        if all ( [ x in kwargs for x in [ 'kernel_dim2 ' , 'kernel_dim3 ' ] ] ) : 
            kernel_size = ( args [ 2 ] , 
            kwargs . pop ( 'kernel_dim2 ' ) , 
            kwargs . pop ( 'kernel_dim3 ' ) ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'kernel_dim* ' ) ) 
    elif len ( args ) == 2 : 
        if all ( [ x in kwargs for x in [ 'kernel_dim1 ' , 'kernel_dim2 ' , 'kernel_dim3 ' ] ] ) : 
            kernel_size = ( kwargs . pop ( 'kernel_dim1 ' ) , 
            kwargs . pop ( 'kernel_dim2 ' ) , 
            kwargs . pop ( 'kernel_dim3 ' ) ) 
            args = [ args [ 0 ] , args [ 1 ] , kernel_size ] 
            converted . append ( ( 'kernel_size ' , 'kernel_dim* ' ) ) 
    elif len ( args ) == 1 : 
        if all ( [ x in kwargs for x in [ 'kernel_dim1 ' , 'kernel_dim2 ' , 'kernel_dim3 ' ] ] ) : 
            kernel_size = ( kwargs . pop ( 'kernel_dim1 ' ) , 
            kwargs . pop ( 'kernel_dim2 ' ) , 
            kwargs . pop ( 'kernel_dim3 ' ) ) 
            kwargs [ 'kernel_size ' ] = kernel_size 
            converted . append ( ( 'kernel_size ' , 'nb_row/nb_col ' ) ) 
    return args , kwargs , converted 
legacy_conv3d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'filters ' , 'kernel_size ' ] , 
conversions = [ ( 'nb_filter ' , 'filters ' ) , 
( 'subsample ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'W_constraint ' , 'kernel_constraint ' ) , 
( 'b_constraint ' , 'bias_constraint ' ) , 
( 'bias ' , 'use_bias ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } , 
preprocessor = conv3d_args_preprocessor ) 
def batchnorm_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if len ( args ) > 1 : 
        raise TypeError ( 'The `BatchNormalization` layer ' 
        'does not accept positional arguments. ' 
        'Use keyword arguments instead. ' ) 
    if 'mode ' in kwargs : 
        value = kwargs . pop ( 'mode ' ) 
        if value != 0 : 
            raise TypeError ( 'The `mode` argument of `BatchNormalization` ' 
            'no longer exists. `mode=1` and `mode=2` ' 
            'are no longer supported. ' ) 
        converted . append ( ( 'mode ' , None ) ) 
    return args , kwargs , converted 
def convlstm2d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if 'forget_bias_init ' in kwargs : 
        value = kwargs . pop ( 'forget_bias_init ' ) 
        if value == 'one ' : 
            kwargs [ 'unit_forget_bias ' ] = True 
            converted . append ( ( 'forget_bias_init ' , 'unit_forget_bias ' ) ) 
        else : 
            warnings . warn ( 'The `forget_bias_init` argument ' 
            'has been ignored. Use `unit_forget_bias=True` ' 
            'instead to initialize with ones. ' , stacklevel = 3 ) 
    args , kwargs , _converted = conv2d_args_preprocessor ( args , kwargs ) 
    return args , kwargs , converted + _converted 
legacy_convlstm2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'filters ' , 'kernel_size ' ] , 
conversions = [ ( 'nb_filter ' , 'filters ' ) , 
( 'subsample ' , 'strides ' ) , 
( 'border_mode ' , 'padding ' ) , 
( 'dim_ordering ' , 'data_format ' ) , 
( 'init ' , 'kernel_initializer ' ) , 
( 'inner_init ' , 'recurrent_initializer ' ) , 
( 'W_regularizer ' , 'kernel_regularizer ' ) , 
( 'U_regularizer ' , 'recurrent_regularizer ' ) , 
( 'b_regularizer ' , 'bias_regularizer ' ) , 
( 'inner_activation ' , 'recurrent_activation ' ) , 
( 'dropout_W ' , 'dropout ' ) , 
( 'dropout_U ' , 'recurrent_dropout ' ) , 
( 'bias ' , 'use_bias ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } , 
preprocessor = convlstm2d_args_preprocessor ) 
legacy_batchnorm_support = generate_legacy_interface ( 
allowed_positional_args = [ ] , 
conversions = [ ( 'beta_init ' , 'beta_initializer ' ) , 
( 'gamma_init ' , 'gamma_initializer ' ) ] , 
preprocessor = batchnorm_args_preprocessor ) 
def zeropadding2d_args_preprocessor ( args , kwargs ) : 
    converted = [ ] 
    if 'padding ' in kwargs and isinstance ( kwargs [ 'padding ' ] , dict ) : 
        if set ( kwargs [ 'padding ' ] . keys ( ) ) <= { 'top_pad ' , 'bottom_pad ' , 
        'left_pad ' , 'right_pad ' } : 
            top_pad = kwargs [ 'padding ' ] . get ( 'top_pad ' , 0 ) 
            bottom_pad = kwargs [ 'padding ' ] . get ( 'bottom_pad ' , 0 ) 
            left_pad = kwargs [ 'padding ' ] . get ( 'left_pad ' , 0 ) 
            right_pad = kwargs [ 'padding ' ] . get ( 'right_pad ' , 0 ) 
            kwargs [ 'padding ' ] = ( ( top_pad , bottom_pad ) , ( left_pad , right_pad ) ) 
            warnings . warn ( 'The `padding` argument in the Keras 2 API no longer ' 
            'accepts dict types. You can now input argument as: ' 
            '`padding=(top_pad, bottom_pad, left_pad, right_pad)`. ' , 
            stacklevel = 3 ) 
    elif len ( args ) == 2 and isinstance ( args [ 1 ] , dict ) : 
        if set ( args [ 1 ] . keys ( ) ) <= { 'top_pad ' , 'bottom_pad ' , 
        'left_pad ' , 'right_pad ' } : 
            top_pad = args [ 1 ] . get ( 'top_pad ' , 0 ) 
            bottom_pad = args [ 1 ] . get ( 'bottom_pad ' , 0 ) 
            left_pad = args [ 1 ] . get ( 'left_pad ' , 0 ) 
            right_pad = args [ 1 ] . get ( 'right_pad ' , 0 ) 
            args = ( args [ 0 ] , ( ( top_pad , bottom_pad ) , ( left_pad , right_pad ) ) ) 
            warnings . warn ( 'The `padding` argument in the Keras 2 API no longer ' 
            'accepts dict types. You can now input argument as: ' 
            '`padding=((top_pad, bottom_pad), (left_pad, right_pad))` ' , 
            stacklevel = 3 ) 
    return args , kwargs , converted 
legacy_zeropadding2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'padding ' ] , 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } , 
preprocessor = zeropadding2d_args_preprocessor ) 
legacy_zeropadding3d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'padding ' ] , 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_cropping2d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'cropping ' ] , 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_cropping3d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'cropping ' ] , 
conversions = [ ( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_spatialdropout1d_support = generate_legacy_interface ( 
allowed_positional_args = [ 'rate ' ] , 
conversions = [ ( 'p ' , 'rate ' ) ] ) 
legacy_spatialdropoutNd_support = generate_legacy_interface ( 
allowed_positional_args = [ 'rate ' ] , 
conversions = [ ( 'p ' , 'rate ' ) , 
( 'dim_ordering ' , 'data_format ' ) ] , 
value_conversions = { 'dim_ordering ' : { 'tf ' : 'channels_last ' , 
'th ' : 'channels_first ' , 
'default ' : None } } ) 
legacy_lambda_support = generate_legacy_interface ( 
allowed_positional_args = [ 'function ' , 'output_shape ' ] ) 


try : 
    from keras_applications import resnet_v2 
except : 
    resnet_v2 = None 
@ keras_modules_injection 
def ResNet50V2 ( * args , ** kwargs ) : 
    return resnet_v2 . ResNet50V2 ( * args , ** kwargs ) 
@ keras_modules_injection 
def ResNet101V2 ( * args , ** kwargs ) : 
    return resnet_v2 . ResNet101V2 ( * args , ** kwargs ) 
@ keras_modules_injection 
def ResNet152V2 ( * args , ** kwargs ) : 
    return resnet_v2 . ResNet152V2 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return resnet_v2 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return resnet_v2 . preprocess_input ( * args , ** kwargs ) 

_DISABLE_TRACKING = threading . local ( ) 
_DISABLE_TRACKING . value = False 
def disable_tracking ( func ) : 
    def wrapped_fn ( * args , ** kwargs ) : 
        global _DISABLE_TRACKING 
        prev_value = _DISABLE_TRACKING . value 
        _DISABLE_TRACKING . value = True 
        out = func ( * args , ** kwargs ) 
        _DISABLE_TRACKING . value = prev_value 
        return out 
    return wrapped_fn 
class Layer ( object ) : 
    def __init__ ( self , ** kwargs ) : 
        self . input_spec = None 
        self . supports_masking = False 
        self . stateful = False 

def softmax ( x , axis = - 1 ) : 
    ndim = K . ndim ( x ) 
    if ndim == 2 : 
        return K . softmax ( x ) 
    elif ndim > 2 : 
        e = K . exp ( x - K . max ( x , axis = axis , keepdims = True ) ) 
        s = K . sum ( e , axis = axis , keepdims = True ) 
        return e / s 
    else : 
        raise ValueError ( 'Cannot apply softmax to a tensor that is 1D. ' 
        'Received input: %s ' % x ) 
def elu ( x , alpha = 1.0 ) : 
    return K . elu ( x , alpha ) 
def selu ( x ) : 
    alpha = 1.6732632423543772848170429916717 
    scale = 1.0507009873554804934193349852946 
    return scale * K . elu ( x , alpha ) 
def softplus ( x ) : 
    return K . softplus ( x ) 
def softsign ( x ) : 
    return K . softsign ( x ) 
def relu ( x , alpha = 0. , max_value = None , threshold = 0. ) : 
    return K . relu ( x , alpha = alpha , max_value = max_value , threshold = threshold ) 
def tanh ( x ) : 
    return K . tanh ( x ) 
def sigmoid ( x ) : 
    return K . sigmoid ( x ) 
def hard_sigmoid ( x ) : 
    return K . hard_sigmoid ( x ) 
def exponential ( x ) : 
    return K . exp ( x ) 
def linear ( x ) : 
    return x 
def serialize ( activation ) : 
    return activation . __name__ 
def deserialize ( name , custom_objects = None ) : 
    return deserialize_keras_object ( 
    name , 
    module_objects = globals ( ) , 
    custom_objects = custom_objects , 
    printable_module_name = 'activation function ' ) 
def get ( identifier ) : 
    if identifier is None : 
        return linear 
    if isinstance ( identifier , six . string_types ) : 
        identifier = str ( identifier ) 
        return deserialize ( identifier ) 
    elif callable ( identifier ) : 
        if isinstance ( identifier , Layer ) : 
            warnings . warn ( 
            'Do not pass a layer instance (such as {identifier}) as the ' 
            'activation argument of another layer. Instead, advanced ' 
            'activation layers should be used just like any other ' 
            'layer in a model. ' . format ( 
            identifier = identifier . __class__ . __name__ ) ) 
        return identifier 
    else : 
        raise ValueError ( 'Could not interpret ' 
        'activation function identifier: ' , identifier ) 

def normalize_tuple ( value , n , name ) : 
    if isinstance ( value , int ) : 
        return ( value , ) * n 
    else : 
        try : 
            value_tuple = tuple ( value ) 
        except TypeError : 
            raise ValueError ( 'The `{}` argument must be a tuple of {} ' 
            'integers. Received: {} ' . format ( name , n , value ) ) 
        if len ( value_tuple ) != n : 
            raise ValueError ( 'The `{}` argument must be a tuple of {} ' 
            'integers. Received: {} ' . format ( name , n , value ) ) 
        for single_value in value_tuple : 
            try : 
                int ( single_value ) 
            except ValueError : 
                raise ValueError ( 'The `{}` argument must be a tuple of {} ' 
                'integers. Received: {} including element {} ' 
                'of type {} ' . format ( name , n , value , single_value , 
                type ( single_value ) ) ) 
    return value_tuple 
def normalize_padding ( value ) : 
    padding = value . lower ( ) 
    allowed = { 'valid ' , 'same ' , 'causal ' } 
    if K . backend ( ) == 'theano ' : 
        allowed . add ( 'full ' ) 
    if padding not in allowed : 
        raise ValueError ( 'The `padding` argument must be one of "valid", "same" ' 
        '(or "causal" for Conv1D). Received: {} ' . format ( padding ) ) 
    return padding 
def convert_kernel ( kernel ) : 
    kernel = np . asarray ( kernel ) 
    if not 3 <= kernel . ndim <= 5 : 
        raise ValueError ( 'Invalid kernel shape: ' , kernel . shape ) 
    slices = [ slice ( None , None , - 1 ) for _ in range ( kernel . ndim ) ] 
    no_flip = ( slice ( None , None ) , slice ( None , None ) ) 
    slices [ - 2 : ] = no_flip 
    return np . copy ( kernel [ tuple ( slices ) ] ) 
def conv_output_length ( input_length , filter_size , 
padding , stride , dilation = 1 ) : 
    if input_length is None : 
        return None 
    assert padding in { 'same ' , 'valid ' , 'full ' , 'causal ' } 
    dilated_filter_size = ( filter_size - 1 ) * dilation + 1 
    if padding == 'same ' : 
        output_length = input_length 
    elif padding == 'valid ' : 
        output_length = input_length - dilated_filter_size + 1 
    elif padding == 'causal ' : 
        output_length = input_length 
    elif padding == 'full ' : 
        output_length = input_length + dilated_filter_size - 1 
    return ( output_length + stride - 1 ) // stride 
def conv_input_length ( output_length , filter_size , padding , stride ) : 
    if output_length is None : 
        return None 
    assert padding in { 'same ' , 'valid ' , 'full ' } 
    if padding == 'same ' : 
        pad = filter_size // 2 
    elif padding == 'valid ' : 
        pad = 0 
    elif padding == 'full ' : 
        pad = filter_size - 1 
    return ( output_length - 1 ) * stride - 2 * pad + filter_size 
def deconv_length ( dim_size , stride_size , kernel_size , padding , 
output_padding , dilation = 1 ) : 
    assert padding in { 'same ' , 'valid ' , 'full ' } 
    if dim_size is None : 
        return None 


try : 
    import theano . sparse as th_sparse_module 
except ImportError : 
    th_sparse_module = None 
try : 
    from theano . tensor . nnet . nnet import softsign as T_softsign 
except ImportError : 
    from theano . sandbox . softsign import softsign as T_softsign 

@ keras_modules_injection 
def DenseNet121 ( * args , ** kwargs ) : 
    return densenet . DenseNet121 ( * args , ** kwargs ) 
@ keras_modules_injection 
def DenseNet169 ( * args , ** kwargs ) : 
    return densenet . DenseNet169 ( * args , ** kwargs ) 
@ keras_modules_injection 
def DenseNet201 ( * args , ** kwargs ) : 
    return densenet . DenseNet201 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return densenet . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return densenet . preprocess_input ( * args , ** kwargs ) 


class _Merge ( Layer ) : 
    def __init__ ( self , ** kwargs ) : 
        super ( _Merge , self ) . __init__ ( ** kwargs ) 
        self . supports_masking = True 
    def _merge_function ( self , inputs ) : 
        raise NotImplementedError 
    def _compute_elemwise_op_output_shape ( self , shape1 , shape2 ) : 
        if None in [ shape1 , shape2 ] : 
            return None 
        elif len ( shape1 ) < len ( shape2 ) : 
            return self . _compute_elemwise_op_output_shape ( shape2 , shape1 ) 
        elif not shape2 : 
            return shape1 
        output_shape = list ( shape1 [ : - len ( shape2 ) ] ) 
        for i , j in zip ( shape1 [ - len ( shape2 ) : ] , shape2 ) : 
            if i is None or j is None : 
                output_shape . append ( None ) 
            elif i == 1 : 
                output_shape . append ( j ) 
            elif j == 1 : 
                output_shape . append ( i ) 
            else : 
                if i != j : 
                    raise ValueError ( 'Operands could not be broadcast ' 
                    'together with shapes ' + 
                    str ( shape1 ) + '' + str ( shape2 ) ) 
                output_shape . append ( i ) 
        return tuple ( output_shape ) 
    def build ( self , input_shape ) : 

try : 
    from tensorflow . python . lib . io import file_io as tf_file_io 
except ImportError : 
    tf_file_io = None 
try : 
    from unittest . mock import patch , Mock , MagicMock 
except : 
    from mock import patch , Mock , MagicMock 
def get_test_data ( num_train = 1000 , num_test = 500 , input_shape = ( 10 , ) , 
output_shape = ( 2 , ) , 
classification = True , num_classes = 2 ) : 
    samples = num_train + num_test 
    if classification : 
        y = np . random . randint ( 0 , num_classes , size = ( samples , ) ) 
        X = np . zeros ( ( samples , ) + input_shape , dtype = np . float32 ) 
        for i in range ( samples ) : 
            X [ i ] = np . random . normal ( loc = y [ i ] , scale = 0.7 , size = input_shape ) 
    else : 
        y_loc = np . random . random ( ( samples , ) ) 
        X = np . zeros ( ( samples , ) + input_shape , dtype = np . float32 ) 
        y = np . zeros ( ( samples , ) + output_shape , dtype = np . float32 ) 
        for i in range ( samples ) : 
            X [ i ] = np . random . normal ( loc = y_loc [ i ] , scale = 0.7 , size = input_shape ) 
            y [ i ] = np . random . normal ( loc = y_loc [ i ] , scale = 0.7 , size = output_shape ) 
    return ( X [ : num_train ] , y [ : num_train ] ) , ( X [ num_train : ] , y [ num_train : ] ) 
def layer_test ( layer_cls , kwargs = { } , input_shape = None , input_dtype = None , 
input_data = None , expected_output = None , 
expected_output_dtype = None , fixed_batch_size = False ) : 

text_to_word_sequence = text . text_to_word_sequence 
one_hot = text . one_hot 
hashing_trick = text . hashing_trick 
Tokenizer = text . Tokenizer 
tokenizer_from_json = text . tokenizer_from_json 

if K . backend ( ) == 'tensorflow ' and not K . tensorflow_backend . _is_tf_1 ( ) : 
    from . tensorboard_v2 import TensorBoard 
else : 
    from . tensorboard_v1 import TensorBoard 


class InputLayer ( Layer ) : 
    @ interfaces . legacy_input_support 
    def __init__ ( self , input_shape = None , batch_size = None , 
    batch_input_shape = None , 
    dtype = None , input_tensor = None , sparse = False , name = None ) : 
        if not name : 
            prefix = 'input ' 
            name = prefix + '_ ' + str ( K . get_uid ( prefix ) ) 
        super ( InputLayer , self ) . __init__ ( dtype = dtype , name = name ) 
        self . trainable = False 
        self . built = True 
        self . sparse = sparse 
        self . supports_masking = True 
        if input_shape and batch_input_shape : 
            raise ValueError ( 'Only provide the input_shape OR ' 
            'batch_input_shape argument to ' 
            'InputLayer, not both at the same time. ' ) 
        if input_tensor is not None and batch_input_shape is None : 


@ keras_modules_injection 
def InceptionV3 ( * args , ** kwargs ) : 
    return inception_v3 . InceptionV3 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return inception_v3 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return inception_v3 . preprocess_input ( * args , ** kwargs ) 

def to_categorical ( y , num_classes = None , dtype = 'float32 ' ) : 
    y = np . array ( y , dtype = 'int ' ) 
    input_shape = y . shape 
    if input_shape and input_shape [ - 1 ] == 1 and len ( input_shape ) > 1 : 
        input_shape = tuple ( input_shape [ : - 1 ] ) 
    y = y . ravel ( ) 
    if not num_classes : 
        num_classes = np . max ( y ) + 1 
    n = y . shape [ 0 ] 
    categorical = np . zeros ( ( n , num_classes ) , dtype = dtype ) 
    categorical [ np . arange ( n ) , y ] = 1 
    output_shape = input_shape + ( num_classes , ) 
    categorical = np . reshape ( categorical , output_shape ) 
    return categorical 
def normalize ( x , axis = - 1 , order = 2 ) : 
    l2 = np . atleast_1d ( np . linalg . norm ( x , order , axis ) ) 
    l2 [ l2 == 0 ] = 1 
    return x / np . expand_dims ( l2 , axis ) 


class Reduction ( object ) : 
    NONE = 'none ' 
    SUM = 'sum ' 
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size ' 
    @ classmethod 
    def all ( cls ) : 
        return ( cls . NONE , cls . SUM , cls . SUM_OVER_BATCH_SIZE ) 
    @ classmethod 
    def validate ( cls , key ) : 
        if key not in cls . all ( ) : 
            raise ValueError ( 'Invalid Reduction Key %s. ' % key ) 
def squeeze_or_expand_dimensions ( y_pred , y_true = None , sample_weight = None ) : 
    if y_true is not None : 
        y_pred_rank = K . ndim ( y_pred ) 
        y_pred_shape = K . int_shape ( y_pred ) 
        y_true_rank = K . ndim ( y_true ) 
        y_true_shape = K . int_shape ( y_true ) 
        if ( y_pred_rank - y_true_rank == 1 ) and ( y_pred_shape [ - 1 ] == 1 ) : 
            y_pred = K . squeeze ( y_pred , - 1 ) 
        elif ( y_true_rank - y_pred_rank == 1 ) and ( y_true_shape [ - 1 ] == 1 ) : 
            y_true = K . squeeze ( y_true , - 1 ) 
    if sample_weight is None : 
        return y_pred , y_true 
    y_pred_rank = K . ndim ( y_pred ) 
    weights_rank = K . ndim ( sample_weight ) 
    if weights_rank != 0 : 
        if y_pred_rank == 0 and weights_rank == 1 : 
            y_pred = K . expand_dims ( y_pred , - 1 ) 
        elif weights_rank - y_pred_rank == 1 : 
            sample_weight = K . squeeze ( sample_weight , - 1 ) 
        elif y_pred_rank - weights_rank == 1 : 
            sample_weight = K . expand_dims ( sample_weight , - 1 ) 
    return y_pred , y_true , sample_weight 
def _num_elements ( losses ) : 
    with K . name_scope ( 'num_elements ' ) as scope : 
        return K . cast ( K . size ( losses , name = scope ) , losses . dtype ) 
def reduce_weighted_loss ( weighted_losses , reduction = Reduction . SUM_OVER_BATCH_SIZE ) : 
    if reduction == Reduction . NONE : 
        loss = weighted_losses 
    else : 
        loss = K . sum ( weighted_losses ) 
        if reduction == Reduction . SUM_OVER_BATCH_SIZE : 
            loss = loss / _num_elements ( weighted_losses ) 
    return loss 
def broadcast_weights ( values , sample_weight ) : 


class TensorBoard ( tf . keras . callbacks . TensorBoard ) : 
    def __init__ ( self , log_dir = './logs ' , 
    histogram_freq = 0 , 
    batch_size = None , 
    write_graph = True , 
    write_grads = False , 
    write_images = False , 
    embeddings_freq = 0 , 
    embeddings_layer_names = None , 
    embeddings_metadata = None , 
    embeddings_data = None , 
    update_freq = 'epoch ' , 
    ** kwargs ) : 
        if batch_size is not None : 
            warnings . warn ( 'The TensorBoard callback `batch_size` argument ' 
            '(for histogram computation) ' 
            'is deprecated with TensorFlow 2.0. ' 
            'It will be ignored. ' ) 
        if write_grads : 
            warnings . warn ( 'The TensorBoard callback does not support ' 
            'gradients display when using TensorFlow 2.0. ' 
            'The `write_grads` argument is ignored. ' ) 
        if ( embeddings_freq or embeddings_layer_names or 
        embeddings_metadata or embeddings_data ) : 
            warnings . warn ( 'The TensorBoard callback does not support ' 
            'embeddings display when using TensorFlow 2.0. ' 
            'Embeddings-related arguments are ignored. ' ) 
        super ( TensorBoard , self ) . __init__ ( 
        log_dir = log_dir , 
        histogram_freq = histogram_freq , 
        write_graph = write_graph , 
        write_images = write_images , 
        update_freq = update_freq , 
        ** kwargs ) 
    def set_model ( self , model ) : 
        model . run_eagerly = False 
        super ( TensorBoard , self ) . set_model ( model ) 

class Initializer ( object ) : 
    def __call__ ( self , shape , dtype = None ) : 
        raise NotImplementedError 
    def get_config ( self ) : 
        return { } 
    @ classmethod 
    def from_config ( cls , config ) : 
        if 'dtype ' in config : 

try : 
    import h5py 
except ImportError : 
    h5py = None 
class Sequential ( Model ) : 
    def __init__ ( self , layers = None , name = None ) : 
        super ( Sequential , self ) . __init__ ( name = name ) 
        self . _build_input_shape = None 

def load_data ( ) : 
    dirname = 'cifar-10-batches-py ' 
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz ' 
    path = get_file ( dirname , origin = origin , untar = True ) 
    num_train_samples = 50000 
    x_train = np . empty ( ( num_train_samples , 3 , 32 , 32 ) , dtype = 'uint8 ' ) 
    y_train = np . empty ( ( num_train_samples , ) , dtype = 'uint8 ' ) 
    for i in range ( 1 , 6 ) : 
        fpath = os . path . join ( path , 'data_batch_ ' + str ( i ) ) 
        ( x_train [ ( i - 1 ) * 10000 : i * 10000 , : , : , : ] , 
        y_train [ ( i - 1 ) * 10000 : i * 10000 ] ) = load_batch ( fpath ) 
    fpath = os . path . join ( path , 'test_batch ' ) 
    x_test , y_test = load_batch ( fpath ) 
    y_train = np . reshape ( y_train , ( len ( y_train ) , 1 ) ) 
    y_test = np . reshape ( y_test , ( len ( y_test ) , 1 ) ) 
    if K . image_data_format ( ) == 'channels_last ' : 
        x_train = x_train . transpose ( 0 , 2 , 3 , 1 ) 
        x_test = x_test . transpose ( 0 , 2 , 3 , 1 ) 
    return ( x_train , y_train ) , ( x_test , y_test ) 

@ keras_modules_injection 
def MobileNet ( * args , ** kwargs ) : 
    return mobilenet . MobileNet ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return mobilenet . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return mobilenet . preprocess_input ( * args , ** kwargs ) 



@ keras_modules_injection 
def NASNetMobile ( * args , ** kwargs ) : 
    return nasnet . NASNetMobile ( * args , ** kwargs ) 
@ keras_modules_injection 
def NASNetLarge ( * args , ** kwargs ) : 
    return nasnet . NASNetLarge ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return nasnet . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return nasnet . preprocess_input ( * args , ** kwargs ) 

try : 
    import h5py 
    HDF5_OBJECT_HEADER_LIMIT = 64512 
except ImportError : 
    h5py = None 
try : 
    from tensorflow . python . lib . io import file_io as tf_file_io 
except ImportError : 
    tf_file_io = None 
try : 
    getargspec = inspect . getfullargspec 
except AttributeError : 




if backend ( ) == 'theano ' : 
    from . load_backend import pattern_broadcast 
elif backend ( ) == 'tensorflow ' : 
    from . load_backend import clear_session 
    from . load_backend import manual_variable_initialization 
    from . load_backend import get_session 
    from . load_backend import set_session 
elif backend ( ) == 'cntk ' : 
    from . load_backend import clear_session 

@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return imagenet_utils . decode_predictions ( 
    * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return imagenet_utils . preprocess_input ( * args , ** kwargs ) 

def load_data ( path = 'imdb.npz ' , num_words = None , skip_top = 0 , 
maxlen = None , seed = 113 , 
start_char = 1 , oov_char = 2 , index_from = 3 , ** kwargs ) : 

class MaxoutDense ( Layer ) : 
    def __init__ ( self , output_dim , 
    nb_feature = 4 , 
    init = 'glorot_uniform ' , 
    weights = None , 
    W_regularizer = None , 
    b_regularizer = None , 
    activity_regularizer = None , 
    W_constraint = None , 
    b_constraint = None , 
    bias = True , 
    input_dim = None , 
    ** kwargs ) : 
        warnings . warn ( 'The `MaxoutDense` layer is deprecated ' 
        'and will be removed after 06/2017. ' ) 
        self . output_dim = output_dim 
        self . nb_feature = nb_feature 
        self . init = initializers . get ( init ) 
        self . W_regularizer = regularizers . get ( W_regularizer ) 
        self . b_regularizer = regularizers . get ( b_regularizer ) 
        self . activity_regularizer = regularizers . get ( activity_regularizer ) 
        self . W_constraint = constraints . get ( W_constraint ) 
        self . b_constraint = constraints . get ( b_constraint ) 
        self . bias = bias 
        self . initial_weights = weights 
        self . input_spec = InputSpec ( ndim = 2 ) 
        self . input_dim = input_dim 
        if self . input_dim : 
            kwargs [ 'input_shape ' ] = ( self . input_dim , ) 
        super ( MaxoutDense , self ) . __init__ ( ** kwargs ) 
    def build ( self , input_shape ) : 
        input_dim = input_shape [ 1 ] 
        self . input_spec = InputSpec ( dtype = K . floatx ( ) , 
        shape = ( None , input_dim ) ) 
        self . W = self . add_weight ( shape = ( self . nb_feature , input_dim , self . output_dim ) , 
        initializer = self . init , 
        name = 'W ' , 
        regularizer = self . W_regularizer , 
        constraint = self . W_constraint ) 
        if self . bias : 
            self . b = self . add_weight ( shape = ( self . nb_feature , self . output_dim , ) , 
            initializer = 'zero ' , 
            name = 'b ' , 
            regularizer = self . b_regularizer , 
            constraint = self . b_constraint ) 
        else : 
            self . b = None 
        if self . initial_weights is not None : 
            self . set_weights ( self . initial_weights ) 
            del self . initial_weights 
        self . built = True 
    def compute_output_shape ( self , input_shape ) : 
        assert input_shape and len ( input_shape ) == 2 
        return ( input_shape [ 0 ] , self . output_dim ) 
    def call ( self , x ) : 


def standardize_single_array ( x ) : 
    if x is None : 
        return None 
    elif K . is_tensor ( x ) : 
        shape = K . int_shape ( x ) 
        if shape is None or shape [ 0 ] is None : 
            raise ValueError ( 
            'When feeding symbolic tensors to a model, we expect the ' 
            'tensors to have a static batch size. ' 
            'Got tensor with shape: %s ' % str ( shape ) ) 
        return x 
    elif x . ndim == 1 : 
        x = np . expand_dims ( x , 1 ) 
    return x 
def standardize_input_data ( data , 
names , 
shapes = None , 
check_batch_axis = True , 
exception_prefix = '' ) : 
    if not names : 
        if data is not None and hasattr ( data , '__len__ ' ) and len ( data ) : 
            raise ValueError ( 'Error when checking model ' + 
            exception_prefix + ': ' 
            'expected no data, but got: ' , data ) 
        return [ ] 
    if data is None : 
        return [ None for _ in range ( len ( names ) ) ] 
    if isinstance ( data , dict ) : 
        try : 
            data = [ 
            data [ x ] . values 
            if data [ x ] . __class__ . __name__ == 'DataFrame ' else data [ x ] 
            for x in names 
            ] 
        except KeyError as e : 
            raise ValueError ( 'No data provided for " '+ e . args [ 0 ] + 
 ' ". Need data '
 ' for each key in: '+ str ( names ) ) 
 elif isinstance ( data , list ) : 
 if isinstance ( data [ 0 ] , list ) : 
 data = [ np . asarray ( d ) for d in data ] 
 elif len ( names ) == 1 and isinstance ( data [ 0 ] , ( float , int ) ) : 
 data = [ np . asarray ( data ) ] 
 else : 
 data = [ 
 x . values if x . __class__ . __name__ == ' DataFrame '
 else x for x in data 
 ] 
 else : 
 data = data . values if data . __class__ . __name__ == ' DataFrame 'else data 
 data = [ data ] 
 data = [ standardize_single_array ( x ) for x in data ] 
 if len ( data ) != len ( names ) : 
 if data and hasattr ( data [ 0 ] , ' shape ') : 
 raise ValueError ( 
 ' Error when checking model '+ exception_prefix + 
 ' : the list of Numpy arrays that you are passing to '
 ' your model is not the size the model expected. '
 ' Expected to see '+ str ( len ( names ) ) + ' array(s), '
 ' but instead got the following list of '+ 
 str ( len ( data ) ) + ' arrays: '+ str ( data ) [ : 200 ] + ' ... ') 
 elif len ( names ) > 1 : 
 raise ValueError ( 
 ' Error when checking model '+ exception_prefix + 
 ' : you are passing a list as input to your model, '
 ' but the model expects a list of '+ str ( len ( names ) ) + 
 ' Numpy arrays instead. '
 ' The list you passed was: '+ str ( data ) [ : 200 ] ) 
 elif len ( data ) == 1 and not hasattr ( data [ 0 ] , ' shape ') : 
 raise TypeError ( ' Error when checking model '+ exception_prefix + 
 ' : data should be a Numpy array, or list/dict of '
 ' Numpy arrays. Found: '+ str ( data ) [ : 200 ] + ' ... ''
EOF

class Regularizer ( object ) : 
    def __call__ ( self , x ) : 
        return 0. 
    @ classmethod 
    def from_config ( cls , config ) : 
        return cls ( ** config ) 
class L1L2 ( Regularizer ) : 
    def __init__ ( self , l1 = 0. , l2 = 0. ) : 
        self . l1 = K . cast_to_floatx ( l1 ) 
        self . l2 = K . cast_to_floatx ( l2 ) 
    def __call__ ( self , x ) : 
        regularization = 0. 
        if self . l1 : 
            regularization += self . l1 * K . sum ( K . abs ( x ) ) 
        if self . l2 : 
            regularization += self . l2 * K . sum ( K . square ( x ) ) 
        return regularization 
    def get_config ( self ) : 
        return { 'l1 ' : float ( self . l1 ) , 
        'l2 ' : float ( self . l2 ) } 

class _CuDNNRNN ( RNN ) : 
    def __init__ ( self , 
    return_sequences = False , 
    return_state = False , 
    go_backwards = False , 
    stateful = False , 
    ** kwargs ) : 
        if K . backend ( ) != 'tensorflow ' : 
            raise RuntimeError ( 'CuDNN RNNs are only available ' 
            'with the TensorFlow backend. ' ) 
        super ( RNN , self ) . __init__ ( ** kwargs ) 
        self . return_sequences = return_sequences 
        self . return_state = return_state 
        self . go_backwards = go_backwards 
        self . stateful = stateful 
        self . supports_masking = False 
        self . input_spec = [ InputSpec ( ndim = 3 ) ] 
        if hasattr ( self . cell . state_size , '__len__ ' ) : 
            state_size = self . cell . state_size 
        else : 
            state_size = [ self . cell . state_size ] 
        self . state_spec = [ InputSpec ( shape = ( None , dim ) ) 
        for dim in state_size ] 
        self . constants_spec = None 
        self . _states = None 
        self . _num_constants = None 
    def _canonical_to_params ( self , weights , biases ) : 
        import tensorflow as tf 
        weights = [ tf . reshape ( x , ( - 1 , ) ) for x in weights ] 
        biases = [ tf . reshape ( x , ( - 1 , ) ) for x in biases ] 
        return tf . concat ( weights + biases , 0 ) 
    def call ( self , inputs , mask = None , training = None , initial_state = None ) : 
        if isinstance ( mask , list ) : 
            mask = mask [ 0 ] 
        if mask is not None : 
            raise ValueError ( 'Masking is not supported for CuDNN RNNs. ' ) 

pad_sequences = sequence . pad_sequences 
make_sampling_table = sequence . make_sampling_table 
skipgrams = sequence . skipgrams 
_remove_long_seq = sequence . _remove_long_seq 

def keras_modules_injection ( base_fun ) : 
    def wrapper ( * args , ** kwargs ) : 
        kwargs [ 'backend ' ] = backend 
        kwargs [ 'layers ' ] = layers 
        kwargs [ 'models ' ] = models 
        kwargs [ 'utils ' ] = utils 
        return base_fun ( * args , ** kwargs ) 
    return wrapper 

def load_data ( path = 'boston_housing.npz ' , test_split = 0.2 , seed = 113 ) : 
    assert 0 <= test_split < 1 
    path = get_file ( 
    path , 
    origin = 'https://s3.amazonaws.com/keras-datasets/boston_housing.npz ' , 
    file_hash = 'f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5 ' ) 
    with np . load ( path , allow_pickle = True ) as f : 
        x = f [ 'x ' ] 
        y = f [ 'y ' ] 
    rng = np . random . RandomState ( seed ) 
    indices = np . arange ( len ( x ) ) 
    rng . shuffle ( indices ) 
    x = x [ indices ] 
    y = y [ indices ] 
    x_train = np . array ( x [ : int ( len ( x ) * ( 1 - test_split ) ) ] ) 
    y_train = np . array ( y [ : int ( len ( x ) * ( 1 - test_split ) ) ] ) 
    x_test = np . array ( x [ int ( len ( x ) * ( 1 - test_split ) ) : ] ) 
    y_test = np . array ( y [ int ( len ( x ) * ( 1 - test_split ) ) : ] ) 
    return ( x_train , y_train ) , ( x_test , y_test ) 

def _get_available_devices ( ) : 
    return K . tensorflow_backend . _get_available_gpus ( ) + [ '/cpu:0 ' ] 
def _normalize_device_name ( name ) : 
    name = '/ ' + ': ' . join ( name . lower ( ) . replace ( '/ ' , '' ) . split ( ': ' ) [ - 2 : ] ) 
    return name 
def multi_gpu_model ( model , gpus = None , cpu_merge = True , cpu_relocation = False ) : 
    if K . backend ( ) != 'tensorflow ' : 
        raise ValueError ( '`multi_gpu_model` is only available ' 
        'with the TensorFlow backend. ' ) 
    available_devices = _get_available_devices ( ) 
    available_devices = [ _normalize_device_name ( name ) 
    for name in available_devices ] 
    if not gpus : 

@ six . add_metaclass ( abc . ABCMeta ) 
class Metric ( Layer ) : 
    def __init__ ( self , name = None , dtype = None , ** kwargs ) : 
        super ( Metric , self ) . __init__ ( name = name , dtype = dtype , ** kwargs ) 
        self . stateful = True 


@ keras_modules_injection 
def Xception ( * args , ** kwargs ) : 
    return xception . Xception ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return xception . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return xception . preprocess_input ( * args , ** kwargs ) 



_GLOBAL_CUSTOM_OBJECTS = { } 
class CustomObjectScope ( object ) : 
    def __init__ ( self , * args ) : 
        self . custom_objects = args 
        self . backup = None 
    def __enter__ ( self ) : 
        self . backup = _GLOBAL_CUSTOM_OBJECTS . copy ( ) 
        for objects in self . custom_objects : 
            _GLOBAL_CUSTOM_OBJECTS . update ( objects ) 
        return self 
    def __exit__ ( self , * args , ** kwargs ) : 
        _GLOBAL_CUSTOM_OBJECTS . clear ( ) 
        _GLOBAL_CUSTOM_OBJECTS . update ( self . backup ) 
def custom_object_scope ( * args ) : 
    return CustomObjectScope ( * args ) 
def get_custom_objects ( ) : 
    return _GLOBAL_CUSTOM_OBJECTS 
def serialize_keras_object ( instance ) : 
    if instance is None : 
        return None 
    if hasattr ( instance , 'get_config ' ) : 
        return { 
        'class_name ' : instance . __class__ . __name__ , 
        'config ' : instance . get_config ( ) 
        } 
    if hasattr ( instance , '__name__ ' ) : 
        return instance . __name__ 
    else : 
        raise ValueError ( 'Cannot serialize ' , instance ) 
def deserialize_keras_object ( identifier , module_objects = None , 
custom_objects = None , 
printable_module_name = 'object ' ) : 
    if identifier is None : 
        return None 
    if isinstance ( identifier , dict ) : 

def normalize_conv ( func ) : 
    def wrapper ( * args , ** kwargs ) : 
        x = args [ 0 ] 
        w = args [ 1 ] 
        if x . ndim == 3 : 
            w = np . flipud ( w ) 
            w = np . transpose ( w , ( 1 , 2 , 0 ) ) 
            if kwargs [ 'data_format ' ] == 'channels_last ' : 
                x = np . transpose ( x , ( 0 , 2 , 1 ) ) 
        elif x . ndim == 4 : 
            w = np . fliplr ( np . flipud ( w ) ) 
            w = np . transpose ( w , ( 2 , 3 , 0 , 1 ) ) 
            if kwargs [ 'data_format ' ] == 'channels_last ' : 
                x = np . transpose ( x , ( 0 , 3 , 1 , 2 ) ) 
        else : 
            w = np . flip ( np . fliplr ( np . flipud ( w ) ) , axis = 2 ) 
            w = np . transpose ( w , ( 3 , 4 , 0 , 1 , 2 ) ) 
            if kwargs [ 'data_format ' ] == 'channels_last ' : 
                x = np . transpose ( x , ( 0 , 4 , 1 , 2 , 3 ) ) 
        dilation_rate = kwargs . pop ( 'dilation_rate ' , 1 ) 
        if isinstance ( dilation_rate , int ) : 
            dilation_rate = ( dilation_rate , ) * ( x . ndim - 2 ) 
        for ( i , d ) in enumerate ( dilation_rate ) : 
            if d > 1 : 
                for j in range ( w . shape [ 2 + i ] - 1 ) : 
                    w = np . insert ( w , 2 * j + 1 , 0 , axis = 2 + i ) 
        y = func ( x , w , ** kwargs ) 
        if kwargs [ 'data_format ' ] == 'channels_last ' : 
            if y . ndim == 3 : 
                y = np . transpose ( y , ( 0 , 2 , 1 ) ) 
            elif y . ndim == 4 : 
                y = np . transpose ( y , ( 0 , 2 , 3 , 1 ) ) 
            else : 
                y = np . transpose ( y , ( 0 , 2 , 3 , 4 , 1 ) ) 
        return y 
    return wrapper 
@ normalize_conv 
def conv ( x , w , padding , data_format ) : 
    y = [ ] 
    for i in range ( x . shape [ 0 ] ) : 
        _y = [ ] 
        for j in range ( w . shape [ 1 ] ) : 
            __y = [ ] 
            for k in range ( w . shape [ 0 ] ) : 
                __y . append ( signal . convolve ( x [ i , k ] , w [ k , j ] , mode = padding ) ) 
            _y . append ( np . sum ( np . stack ( __y , axis = - 1 ) , axis = - 1 ) ) 
        y . append ( _y ) 
    y = np . array ( y ) 
    return y 
@ normalize_conv 
def depthwise_conv ( x , w , padding , data_format ) : 
    y = [ ] 
    for i in range ( x . shape [ 0 ] ) : 
        _y = [ ] 
        for j in range ( w . shape [ 0 ] ) : 
            __y = [ ] 
            for k in range ( w . shape [ 1 ] ) : 
                __y . append ( signal . convolve ( x [ i , j ] , w [ j , k ] , mode = padding ) ) 
            _y . append ( np . stack ( __y , axis = 0 ) ) 
        y . append ( np . concatenate ( _y , axis = 0 ) ) 
    y = np . array ( y ) 
    return y 
def separable_conv ( x , w1 , w2 , padding , data_format ) : 
    x2 = depthwise_conv ( x , w1 , padding = padding , data_format = data_format ) 
    return conv ( x2 , w2 , padding = padding , data_format = data_format ) 
def conv_transpose ( x , w , output_shape , padding , data_format , dilation_rate = 1 ) : 
    if x . ndim == 4 : 
        w = np . fliplr ( np . flipud ( w ) ) 
        w = np . transpose ( w , ( 0 , 1 , 3 , 2 ) ) 
    else : 
        w = np . flip ( np . fliplr ( np . flipud ( w ) ) , axis = 2 ) 
        w = np . transpose ( w , ( 0 , 1 , 2 , 4 , 3 ) ) 
    if isinstance ( dilation_rate , int ) : 
        dilation_rate = ( dilation_rate , ) * ( x . ndim - 2 ) 
    for ( i , d ) in enumerate ( dilation_rate ) : 
        if d > 1 : 
            for j in range ( w . shape [ i ] - 1 ) : 
                w = np . insert ( w , 2 * j + 1 , 0 , axis = i ) 
    return conv ( x , w , padding = padding , data_format = data_format ) 
conv1d = conv 
conv2d = conv 
conv3d = conv 
depthwise_conv2d = depthwise_conv 
separable_conv1d = separable_conv 
separable_conv2d = separable_conv 
conv2d_transpose = conv_transpose 
conv3d_transpose = conv_transpose 
def pool ( x , pool_size , strides , padding , data_format , pool_mode ) : 
    if data_format == 'channels_last ' : 
        if x . ndim == 3 : 
            x = np . transpose ( x , ( 0 , 2 , 1 ) ) 
        elif x . ndim == 4 : 
            x = np . transpose ( x , ( 0 , 3 , 1 , 2 ) ) 
        else : 
            x = np . transpose ( x , ( 0 , 4 , 1 , 2 , 3 ) ) 
    if padding == 'same ' : 
        pad = [ ( 0 , 0 ) , ( 0 , 0 ) ] + [ ( s // 2 , s // 2 ) for s in pool_size ] 
        x = np . pad ( x , pad , 'constant ' , constant_values = - np . inf ) 

def load_data ( label_mode = 'fine ' ) : 
    if label_mode not in [ 'fine ' , 'coarse ' ] : 
        raise ValueError ( '`label_mode` must be one of `"fine"`, `"coarse"`. ' ) 
    dirname = 'cifar-100-python ' 
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz ' 
    path = get_file ( dirname , origin = origin , untar = True ) 
    fpath = os . path . join ( path , 'train ' ) 
    x_train , y_train = load_batch ( fpath , label_key = label_mode + '_labels ' ) 
    fpath = os . path . join ( path , 'test ' ) 
    x_test , y_test = load_batch ( fpath , label_key = label_mode + '_labels ' ) 
    y_train = np . reshape ( y_train , ( len ( y_train ) , 1 ) ) 
    y_test = np . reshape ( y_test , ( len ( y_test ) , 1 ) ) 
    if K . image_data_format ( ) == 'channels_last ' : 
        x_train = x_train . transpose ( 0 , 2 , 3 , 1 ) 
        x_test = x_test . transpose ( 0 , 2 , 3 , 1 ) 
    return ( x_train , y_train ) , ( x_test , y_test ) 

@ keras_modules_injection 
def InceptionResNetV2 ( * args , ** kwargs ) : 
    return inception_resnet_v2 . InceptionResNetV2 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return inception_resnet_v2 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return inception_resnet_v2 . preprocess_input ( * args , ** kwargs ) 

try : 
    import h5py 
except ImportError : 
    h5py = None 
def _clone_functional_model ( model , input_tensors = None ) : 
    if not isinstance ( model , Model ) : 
        raise ValueError ( 'Expected `model` argument ' 
        'to be a `Model` instance, got ' , model ) 
    if isinstance ( model , Sequential ) : 
        raise ValueError ( 'Expected `model` argument ' 
        'to be a functional `Model` instance, ' 
        'got a `Sequential` instance instead: ' , model ) 
    layer_map = { } 

def fit_loop ( model , fit_function , fit_inputs , 
out_labels = None , 
batch_size = None , 
epochs = 100 , 
verbose = 1 , 
callbacks = None , 
val_function = None , 
val_inputs = None , 
shuffle = True , 
initial_epoch = 0 , 
steps_per_epoch = None , 
validation_steps = None , 
validation_freq = 1 ) : 
    do_validation = False 
    if val_function and val_inputs : 
        do_validation = True 
        if ( verbose and fit_inputs and 
        hasattr ( fit_inputs [ 0 ] , 'shape ' ) and hasattr ( val_inputs [ 0 ] , 'shape ' ) ) : 
            print ( 'Train on %d samples, validate on %d samples ' % 
            ( fit_inputs [ 0 ] . shape [ 0 ] , val_inputs [ 0 ] . shape [ 0 ] ) ) 
    if validation_steps : 
        do_validation = True 
        if steps_per_epoch is None : 
            raise ValueError ( 'Can only use `validation_steps` ' 
            'when doing step-wise ' 
            'training, i.e. `steps_per_epoch` ' 
            'must be set. ' ) 
    elif do_validation : 
        if steps_per_epoch : 
            raise ValueError ( 'Must specify `validation_steps` ' 
            'to perform validation ' 
            'when doing step-wise training. ' ) 
    num_train_samples = check_num_samples ( fit_inputs , 
    batch_size = batch_size , 
    steps = steps_per_epoch , 
    steps_name = 'steps_per_epoch ' ) 
    if num_train_samples is not None : 
        index_array = np . arange ( num_train_samples ) 
    model . history = cbks . History ( ) 
    _callbacks = [ cbks . BaseLogger ( stateful_metrics = model . metrics_names [ 1 : ] ) ] 
    if verbose : 
        if steps_per_epoch is not None : 
            count_mode = 'steps ' 
        else : 
            count_mode = 'samples ' 
        _callbacks . append ( 
        cbks . ProgbarLogger ( count_mode , stateful_metrics = model . metrics_names [ 1 : ] ) ) 
    _callbacks += ( callbacks or [ ] ) + [ model . history ] 
    callbacks = cbks . CallbackList ( _callbacks ) 
    out_labels = out_labels or [ ] 

@ keras_modules_injection 
def VGG19 ( * args , ** kwargs ) : 
    return vgg19 . VGG19 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return vgg19 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return vgg19 . preprocess_input ( * args , ** kwargs ) 



def count_params ( weights ) : 
    weight_ids = set ( ) 
    total = 0 
    for w in weights : 
        if id ( w ) not in weight_ids : 
            weight_ids . add ( id ( w ) ) 
            total += int ( K . count_params ( w ) ) 
    return total 
def print_summary ( model , line_length = None , positions = None , print_fn = None ) : 
    if print_fn is None : 
        print_fn = print 
    if model . __class__ . __name__ == 'Sequential ' : 
        sequential_like = True 
    elif not model . _is_graph_network : 


def load_data ( ) : 
    dirname = os . path . join ( 'datasets ' , 'fashion-mnist ' ) 
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/ ' 
    files = [ 'train-labels-idx1-ubyte.gz ' , 'train-images-idx3-ubyte.gz ' , 
    't10k-labels-idx1-ubyte.gz ' , 't10k-images-idx3-ubyte.gz ' ] 
    paths = [ ] 
    for fname in files : 
        paths . append ( get_file ( fname , 
        origin = base + fname , 
        cache_subdir = dirname ) ) 
    with gzip . open ( paths [ 0 ] , 'rb ' ) as lbpath : 
        y_train = np . frombuffer ( lbpath . read ( ) , np . uint8 , offset = 8 ) 
    with gzip . open ( paths [ 1 ] , 'rb ' ) as imgpath : 
        x_train = np . frombuffer ( imgpath . read ( ) , np . uint8 , 
        offset = 16 ) . reshape ( len ( y_train ) , 28 , 28 ) 
    with gzip . open ( paths [ 2 ] , 'rb ' ) as lbpath : 
        y_test = np . frombuffer ( lbpath . read ( ) , np . uint8 , offset = 8 ) 
    with gzip . open ( paths [ 3 ] , 'rb ' ) as imgpath : 
        x_test = np . frombuffer ( imgpath . read ( ) , np . uint8 , 
        offset = 16 ) . reshape ( len ( y_test ) , 28 , 28 ) 
    return ( x_train , y_train ) , ( x_test , y_test ) 

C . set_global_option ( 'align_axis ' , 1 ) 
b_any = any 
py_slice = slice 
dev = C . device . use_default_device ( ) 
if dev . type ( ) == 0 : 
    warnings . warn ( 
    'CNTK backend warning: GPU is not detected. ' 
    'CNTK\'s CPU version is not fully optimized, ' 
    'please run with GPU to get better performance. ' ) 
    # A learning phase is a bool tensor used to run Keras models in 
    # either train mode (learning_phase == 1) or test mode (learning_phase == 0). 
    # LEARNING_PHASE_PLACEHOLDER is the placeholder for dynamic learning phase 
_LEARNING_PHASE_PLACEHOLDER = C . constant ( 
shape = ( ) , dtype = np . float32 , 
value = 1.0 , 
name = '_keras_learning_phase ' ) 
# static learning phase flag, if it is not 0 or 1, we will go with dynamic 
# learning phase tensor. 
_LEARNING_PHASE = - 1 
_UID_PREFIXES = defaultdict ( int ) 
# cntk doesn't support gradient as symbolic op, to hook up with keras model, 

class BaseWrapper ( object ) : 
    def __init__ ( self , build_fn = None , ** sk_params ) : 
        self . build_fn = build_fn 
        self . sk_params = sk_params 
        self . check_params ( sk_params ) 
    def check_params ( self , params ) : 
        legal_params_fns = [ Sequential . fit , Sequential . predict , 
        Sequential . predict_classes , Sequential . evaluate ] 
        if self . build_fn is None : 
            legal_params_fns . append ( self . __call__ ) 
        elif ( not isinstance ( self . build_fn , types . FunctionType ) and 
        not isinstance ( self . build_fn , types . MethodType ) ) : 
            legal_params_fns . append ( self . build_fn . __call__ ) 
        else : 
            legal_params_fns . append ( self . build_fn ) 
        for params_name in params : 
            for fn in legal_params_fns : 
                if has_arg ( fn , params_name ) : 
                    break 
            else : 
                if params_name != 'nb_epoch ' : 
                    raise ValueError ( 
                    '{} is not a legal parameter ' . format ( params_name ) ) 
    def get_params ( self , ** params ) : 
        res = copy . deepcopy ( self . sk_params ) 
        res . update ( { 'build_fn ' : self . build_fn } ) 
        return res 
    def set_params ( self , ** params ) : 
        self . check_params ( params ) 
        self . sk_params . update ( params ) 
        return self 
    def fit ( self , x , y , ** kwargs ) : 
        if self . build_fn is None : 
            self . model = self . __call__ ( ** self . filter_sk_params ( self . __call__ ) ) 
        elif ( not isinstance ( self . build_fn , types . FunctionType ) and 
        not isinstance ( self . build_fn , types . MethodType ) ) : 
            self . model = self . build_fn ( 
            ** self . filter_sk_params ( self . build_fn . __call__ ) ) 
        else : 
            self . model = self . build_fn ( ** self . filter_sk_params ( self . build_fn ) ) 
        if ( losses . is_categorical_crossentropy ( self . model . loss ) and 
        len ( y . shape ) != 2 ) : 
            y = to_categorical ( y ) 
        fit_args = copy . deepcopy ( self . filter_sk_params ( Sequential . fit ) ) 
        fit_args . update ( kwargs ) 
        history = self . model . fit ( x , y , ** fit_args ) 
        return history 
    def filter_sk_params ( self , fn , override = None ) : 
        override = override or { } 
        res = { } 
        for name , value in self . sk_params . items ( ) : 
            if has_arg ( fn , name ) : 
                res . update ( { name : value } ) 
        res . update ( override ) 
        return res 
class KerasClassifier ( BaseWrapper ) : 
    def fit ( self , x , y , sample_weight = None , ** kwargs ) : 
        y = np . array ( y ) 
        if len ( y . shape ) == 2 and y . shape [ 1 ] > 1 : 
            self . classes_ = np . arange ( y . shape [ 1 ] ) 
        elif ( len ( y . shape ) == 2 and y . shape [ 1 ] == 1 ) or len ( y . shape ) == 1 : 
            self . classes_ = np . unique ( y ) 
            y = np . searchsorted ( self . classes_ , y ) 
        else : 
            raise ValueError ( 'Invalid shape for y: ' + str ( y . shape ) ) 
        self . n_classes_ = len ( self . classes_ ) 
        if sample_weight is not None : 
            kwargs [ 'sample_weight ' ] = sample_weight 
        return super ( KerasClassifier , self ) . fit ( x , y , ** kwargs ) 
    def predict ( self , x , ** kwargs ) : 
        kwargs = self . filter_sk_params ( Sequential . predict_classes , kwargs ) 
        proba = self . model . predict ( x , ** kwargs ) 
        if proba . shape [ - 1 ] > 1 : 
            classes = proba . argmax ( axis = - 1 ) 
        else : 
            classes = ( proba > 0.5 ) . astype ( 'int32 ' ) 
        return self . classes_ [ classes ] 
    def predict_proba ( self , x , ** kwargs ) : 
        kwargs = self . filter_sk_params ( Sequential . predict_proba , kwargs ) 
        probs = self . model . predict ( x , ** kwargs ) 

try : 
    import h5py 
except ImportError : 
    h5py = None 
class Network ( Layer ) : 
    @ interfaces . legacy_model_constructor_support 
    def __init__ ( self , * args , ** kwargs ) : 

if K . backend ( ) == 'tensorflow ' : 
    import tensorflow as tf 
def clip_norm ( g , c , n ) : 
    if c <= 0 : 


@ keras_modules_injection 
def ResNet50 ( * args , ** kwargs ) : 
    return resnet50 . ResNet50 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return resnet50 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return resnet50 . preprocess_input ( * args , ** kwargs ) 

@ keras_modules_injection 
def MobileNetV2 ( * args , ** kwargs ) : 
    return mobilenet_v2 . MobileNetV2 ( * args , ** kwargs ) 
@ keras_modules_injection 
def decode_predictions ( * args , ** kwargs ) : 
    return mobilenet_v2 . decode_predictions ( * args , ** kwargs ) 
@ keras_modules_injection 
def preprocess_input ( * args , ** kwargs ) : 
    return mobilenet_v2 . preprocess_input ( * args , ** kwargs ) 


try : 
    import h5py 
    HDF5_OBJECT_HEADER_LIMIT = 64512 
except ImportError : 
    h5py = None 
if sys . version_info [ 0 ] == 3 : 
    import pickle 
else : 
    import cPickle as pickle 
class HDF5Matrix ( object ) : 
    refs = defaultdict ( int ) 
    def __init__ ( self , datapath , dataset , start = 0 , end = None , normalizer = None ) : 
        if h5py is None : 
            raise ImportError ( 'The use of HDF5Matrix requires ' 
            'HDF5 and h5py installed. ' ) 
        if datapath not in list ( self . refs . keys ( ) ) : 
            f = h5py . File ( datapath ) 
            self . refs [ datapath ] = f 
        else : 
            f = self . refs [ datapath ] 
        self . data = f [ dataset ] 
        self . start = start 
        if end is None : 
            self . end = self . data . shape [ 0 ] 
        else : 
            self . end = end 
        self . normalizer = normalizer 
        if self . normalizer is not None : 
            first_val = self . normalizer ( self . data [ 0 : 1 ] ) 
        else : 
            first_val = self . data [ 0 : 1 ] 
        self . _base_shape = first_val . shape [ 1 : ] 
        self . _base_dtype = first_val . dtype 
    def __len__ ( self ) : 
        return self . end - self . start 
    def __getitem__ ( self , key ) : 
        if isinstance ( key , slice ) : 
            start , stop = key . start , key . stop 
            if start is None : 
                start = 0 
            if stop is None : 
                stop = self . shape [ 0 ] 
            if stop + self . start <= self . end : 
                idx = slice ( start + self . start , stop + self . start ) 
            else : 
                raise IndexError 
        elif isinstance ( key , ( int , np . integer ) ) : 
            if key + self . start < self . end : 
                idx = key + self . start 
            else : 
                raise IndexError 
        elif isinstance ( key , np . ndarray ) : 
            if np . max ( key ) + self . start < self . end : 
                idx = ( self . start + key ) . tolist ( ) 
            else : 
                raise IndexError 
        else : 


class Model ( Network ) : 
    @ K . symbolic 
    def compile ( self , optimizer , 
    loss = None , 
    metrics = None , 
    loss_weights = None , 
    sample_weight_mode = None , 
    weighted_metrics = None , 
    target_tensors = None , 
    ** kwargs ) : 
        self . optimizer = optimizers . get ( optimizer ) 
        self . loss = loss or { } 
        self . _compile_metrics = metrics or [ ] 
        self . loss_weights = loss_weights 
        self . sample_weight_mode = sample_weight_mode 
        self . _compile_weighted_metrics = weighted_metrics 

 