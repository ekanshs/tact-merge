from .permutation_spec import (PermutationSpec as PermutationSpec,
                              permutation_spec_from_axes_to_perm as permutation_spec_from_axes_to_perm)

from .weight_matching import (weight_matching as weight_matching)

from .activation_matching import (activation_matching as activation_matching)

from .util import (apply_permutation as apply_permutation,
                  flatten_params as flatten_params,
                  unflatten_params as unflatten_params, 
                  random_permutation as random_permutation, 
                  identity_permutation as identity_permutation,
                  freeze as freeze,
                  unfreeze as unfreeze)

from .util import (
    conv_axes_to_perm as conv_axes_to_perm, 
    conv_no_bias_axes_to_perm as conv_no_bias_axes_to_perm,
    norm_axes_to_perm as norm_axes_to_perm,
    dense_axes_to_perm as dense_axes_to_perm, 
    dense_no_bias_axes_to_perm as dense_no_bias_axes_to_perm 
    )


