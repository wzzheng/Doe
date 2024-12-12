max_frame = 5

tasks_config = {
    'plan': {
        'init': [True] * max_frame,
        'image': [True] * max_frame,
        'desc': [False] * (max_frame-1) + [True],
        'cf': [False] * max_frame,
        'qa': [False] * max_frame,
        'action': [False] * (max_frame-1) + [True],
        'plan': [True] * max_frame
    }, 
    
    'qa': {
        'init': [True] * max_frame,
        'image': [True] * max_frame,
        'desc': [True] * max_frame,
        'cf': [False] * max_frame,
        'qa': [True] * max_frame,
        'action': [True] * max_frame,
        'plan': [True] * max_frame
    }, 

    'counterfactual': {
        'init': [True] * max_frame,
        'image': [True] * max_frame,
        'desc': [True] * max_frame,
        'cf': [True] * max_frame,
        'qa': [False] * max_frame,
        'action': [True] * max_frame,
        'plan': [True] * max_frame
    },

    'Draw': {
        'init': [True] * max_frame,
        'image': [True] * max_frame,
        'desc': [True] * max_frame,
        'cf': [False] * max_frame,
        'qa': [False] * max_frame,
        'action': [True] * max_frame,
        'plan': [True] * max_frame
    }
}

do_generate = {
    'plan': {
        'init': [False] * max_frame,
        'image': [False] * max_frame,
        'desc': [False] * (max_frame-1) + [True],
        'cf': [False] * max_frame,
        'qa': [False] * max_frame,
        'action': [False] * max_frame,
        'plan': [False] * (max_frame-1) + [True]
    }, 

    'qa': {
        'init': [False] * max_frame,
        'image': [False] * max_frame,
        'desc': [True] * max_frame,
        'cf': [False] * max_frame,
        'qa': [True] * max_frame,
        'action': [True] * max_frame,
        'plan': [False] * max_frame
    }, 

    'counterfactual': {
        'init': [False] * max_frame,
        'image': [False] * max_frame,
        'desc': [True] * max_frame,
        'cf': [True] * max_frame,
        'qa': [False] * max_frame,
        'action': [False] * max_frame,
        'plan': [False] * max_frame
    }, 

    'Draw': {
        'init': [False] * max_frame,
        'image': [True] * max_frame,
        'desc': [True] * max_frame,
        'cf': [False] * max_frame,
        'qa': [False] * max_frame,
        'action': [False] * max_frame,
        'plan': [False] * max_frame
    }
}

generation_configs = {
    'plan': {
        'settings': {
            'max_gen_len': 5,
            'temperature': 0.,
            'do_sample': False
        },
        'processor': {
            'text_top_k': 1
        }
    },

    'desc': {
        'settings': {
            'max_gen_len': 1024,
            'temperature': .3,
        },
        'processor': {
            'text_top_k': 3
        }
    },

    'action': {
        'settings': {
            'max_gen_len': 1024,
            'temperature': .5,
        },
        'processor': {
            'text_top_k': 5
        }
    },

    'qa': {
        'settings': {
            'max_gen_len': 1024,
            'temperature': .5,
        },
        'processor': {
            'text_top_k': 5
        }
    },

    'cf': {
        'settings': {
            'max_gen_len': 1024,
            'temperature': .3,
        },
        'processor': {
            'text_top_k': 3
        }
    },

    'image': {
        'settings': {
            'max_gen_len': 1100,
            'temperature': .5,
        },
        'processor': {
            'image_top_k': 200
        }
    }

}