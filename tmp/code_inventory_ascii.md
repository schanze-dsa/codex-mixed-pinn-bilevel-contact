## Appendix A: File Line Counts

| Lines | File |
|---:|---|
| 4275 | `src/train/trainer.py` |
| 1445 | `src/model/pinn_model.py` |
| 1390 | `src/viz/mirror_viz.py` |
| 1134 | `tools/visualize_contact_tightening_plotly.py` |
| 940 | `src/mesh/surface_utils.py` |
| 903 | `main.py` |
| 815 | `src/model/loss_energy.py` |
| 735 | `src/physics/contact/contact_friction_alm.py` |
| 671 | `src/assembly/surfaces.py` |
| 662 | `src/mesh/contact_pairs.py` |
| 629 | `src/inp_io/inp_parser.py` |
| 557 | `src/inp_io/cdb_parser.py` |
| 528 | `src/physics/contact/contact_normal_alm.py` |
| 508 | `src/train/loss_weights.py` |
| 406 | `src/physics/contact/contact_operator.py` |
| 383 | `src/physics/elasticity_residual.py` |
| 331 | `config.yaml` |
| 281 | `src/physics/tightening_model.py` |
| 269 | `tools/cdb_to_json.py` |
| 269 | `tools/audit_preload.py` |
| 265 | `tools/viz_saved_model.py` |
| 258 | `test_model_innovation_hooks.py` |
| 249 | `src/physics/boundary_conditions.py` |
| 241 | `test_trainer_optimization_hooks.py` |
| 216 | `tools/compare_deflections.py` |
| 193 | `tools/bolt_region_metrics.py` |
| 173 | `src/physics/material_lib.py` |
| 160 | `src/mesh/volume_quadrature.py` |
| 92 | `src/train/attach_ties_bcs.py` |
| 91 | `tools/export_from_ckpt.py` |
| 90 | `test_innovation_physics_losses.py` |
| 82 | `test_performance_sync_guards.py` |
| 48 | `src/mesh/interp_utils.py` |
| 43 | `test_forward_mode_normalize_inputs.py` |
| 41 | `src/train/uncertainty_calibration.py` |
| 30 | `src/physics/elasticity_config.py` |
| 26 | `tools/visualize_contact_tightening.py` |
| 0 | `src/viz/__init__.py` |
| 0 | `src/train/__init__.py` |
| 0 | `src/physics/contact/__init__.py` |
| 0 | `src/physics/__init__.py` |
| 0 | `src/model/__init__.py` |
| 0 | `src/mesh/__init__.py` |
| 0 | `src/inp_io/__init__.py` |
| 0 | `src/__init__.py` |

## Appendix B: Symbol Inventory

### `main.py`
- class `_Tee` (L44): `__init__`(L45), `write`(L53), `flush`(L63), `__getattr__`(L67)
- def `_strip_ansi` (L71)
- def `_setup_run_logs` (L76)
- def `_default_saved_model_dir` (L103)
- def `_load_yaml_config` (L118)
- def `_auto_resolve_surface_keys` (L128)
- def `_prepare_config_with_autoguess` (L153)
- def `_run_training` (L854)
- def `main` (L886)

### `src/__init__.py`
- (no top-level class/def)

### `src/assembly/surfaces.py`
- def `_debug` (L32)
- class `ElementFaceRef` (L42): (no methods)
- class `SurfaceDef` (L49): (no methods)
- class `SurfaceResolvers` (L75): (no methods)
- def `to_points` (L93)
- def `sample_surface_by_key` (L118)
- def `_element_surface_to_points` (L162)
- def `_sample_on_polygon` (L275)
- def `_face_normal_and_area` (L299)
- def `_node_surface_to_points` (L328)
- def `_resolve_node_coords` (L353)
- def `_attach_resolvers` (L394)
- def `_as_element_face` (L492)
- def `_is_elset_tuple` (L505)
- def `_empty_Xnw` (L512)
- def `_as_float32` (L518)
- def `_normalize` (L525)
- def `_is_int_like` (L533)
- def `_pca_normals` (L541)
- def `_strip_asm_prefix` (L569)
- def `_check_surface` (L577)
- def `surface_def_to_points` (L585)

### `src/inp_io/__init__.py`
- (no top-level class/def)

### `src/inp_io/cdb_parser.py`
- def `_parse_fixed_width` (L38)
- def `_safe_int` (L51)
- def `_safe_float` (L66)
- def `_expand_range_stream` (L78)
- def `_parse_etblock` (L95)
- def `_parse_nblock` (L118)
- def `_parse_eblock` (L162)
- def `_parse_cmblock` (L220)
- def `_etype_name_from_code` (L252)
- def `_is_contact_component` (L270)
- def `_is_combined_component` (L277)
- def `load_cdb` (L286)
- def `main` (L540)

### `src/inp_io/inp_parser.py`
- class `ElementBlock` (L39): (no methods)
- class `PartMesh` (L46): (no methods)
- class `ContactPair` (L53): (no methods)
- class `TieConstraint` (L60): (no methods)
- class `BoundaryEntry` (L66): (no methods)
- class `InstanceDef` (L70): (no methods)
- class `SetDef` (L75): (no methods)
- class `InteractionProp` (L86): (no methods)
- class `AssemblyModel` (L92): `summary`(L109), `_dequote`(L128), `_aliases`(L132), `_strip_suffix_S`(L144), `expand_elset`(L148), `get_face_nodes`(L196), `finalize`(L234), `get_friction_mu`(L246)
- def `_is_comment_or_empty` (L280)
- def `_extract_param` (L284)
- def `_parse_kw_params` (L289)
- def `_collect_set_items` (L302)
- def `_normalize_surface_items` (L312)
- def `load_inp` (L328)
- def `_print_quick_summary` (L583)
- def `main` (L615)

### `src/mesh/__init__.py`
- (no top-level class/def)

### `src/mesh/contact_pairs.py`
- class `ContactPairSpec` (L41): (no methods)
- class `ContactPairData` (L49): (no methods)
- class `ContactMap` (L84): `concatenate`(L91), `__len__`(L126)
- def `_orthonormal_tangent_basis` (L134)
- def `_fetch_xyz` (L159)
- def `_triangle_gauss_rule` (L174)
- def `_mortar_points_on_surface` (L216)
- def `_compute_area_weights` (L255)
- def `_sorted_node_ids` (L279)
- def `_map_node_ids_to_idx` (L283)
- def `build_contact_pair_data` (L306)
- def `build_contact_pair_data_mortar` (L371)
- def `build_contact_map` (L437)
- def `resample_contact_map` (L563)
- def `guess_surface_key` (L599)

### `src/mesh/interp_utils.py`
- def `interp_bary_tf` (L19)

### `src/mesh/surface_utils.py`
- def `_normalize_surface_key` (L32)
- class `TriSurface` (L83): `__len__`(L98)
- def `_ordered_unique` (L154)
- def `_normalize_etype_conn` (L169)
- def `_face_map_for_type` (L187)
- def `_expand_elset_ids_fallback` (L205)
- def `_emit_tris_from_face` (L234)
- def `_plane_basis` (L254)
- def `_convex_hull_indices` (L273)
- def `_order_contact_nodes` (L300)
- def `resolve_surface_to_tris` (L316)
- def `triangulate_part_boundary` (L588)
- def `_fetch_xyz` (L695)
- def `compute_tri_geometry` (L709)
- def `sample_points_on_surface` (L740)
- def `_closest_pt_on_triangle` (L771)
- def `project_points_onto_surface` (L822)
- def `_coord_provider_for_ts` (L893)
- def `build_contact_surfaces` (L904)

### `src/mesh/volume_quadrature.py`
- def `build_volume_points` (L44)
- def `_volume_points_for_part` (L93)
- def `_centroid_weight_c3d4_block` (L126)
- def `_centroid_weight_c3d8_block` (L142)

### `src/model/__init__.py`
- (no top-level class/def)

### `src/model/pinn_model.py`
- class `FourierConfig` (L43): (no methods)
- class `EncoderConfig` (L50): (no methods)
- class `FieldConfig` (L58): (no methods)
- class `ModelConfig` (L111): (no methods)
- def `_get_activation` (L123)
- def `_maybe_mixed_precision` (L135)
- class `GaussianFourierFeatures` (L148): `__init__`(L157), `build`(L174), `call`(L185), `out_dim`(L200)
- class `FiniteSpectralFeatures` (L207): `__init__`(L210), `call`(L222), `out_dim`(L247)
- class `MLP` (L256): `__init__`(L259), `call`(L294)
- class `GraphConvLayer` (L304): `__init__`(L307), `call`(L327)
- def `_build_knn_graph` (L413)
- def `_knn_to_adj` (L513)
- class `ParamEncoder` (L548): `__init__`(L550), `call`(L561), `_normalize_dim`(L568)
- class `DisplacementNet` (L595): `__init__`(L601), `set_node_semantic_features`(L844), `prebuild_adjacency`(L857), `call`(L883), `set_global_graph`(L1202), `set_contact_residual_hint`(L1213), `_sample_route_alpha`(L1223)
- class `DisplacementModel` (L1246): `__init__`(L1255), `_normalize_inputs`(L1269), `_u_fn_compiled`(L1312), `_u_fn_pointwise_compiled`(L1328), `u_fn`(L1333), `u_fn_pointwise`(L1344), `_us_fn_compiled`(L1358), `_us_fn_pointwise_compiled`(L1373), `us_fn`(L1380), `us_fn_pointwise`(L1388), `_uvar_fn_compiled`(L1402), `uvar_fn`(L1409)
- def `create_displacement_model` (L1416)

### `src/physics/__init__.py`
- (no top-level class/def)

### `src/physics/boundary_conditions.py`
- class `BoundaryConfig` (L46): (no methods)
- class `BoundaryPenalty` (L57): `__init__`(L60), `build_from_numpy`(L94), `reset_for_new_batch`(L138), `build`(L142), `energy`(L159), `residual`(L203), `update_multipliers`(L226), `set_alpha`(L243), `multiply_weights`(L246)

### `src/physics/contact/__init__.py`
- (no top-level class/def)

### `src/physics/contact/contact_friction_alm.py`
- class `FrictionALMConfig` (L90): (no methods)
- class `FrictionContactALM` (L113): `__init__`(L130), `link_normal`(L171), `build_from_numpy`(L175), `snapshot_state`(L241), `restore_state`(L252), `last_slip`(L272), `capture_reference`(L276), `commit_reference`(L288), `reset_reference`(L294), `build_from_cat`(L298), `_absolute_slip_t`(L318), `_relative_slip_t`(L356), `_effective_normal_pressure`(L375), `energy`(L398), `residual`(L568), `update_multipliers`(L641), `set_mu_t`(L681), `set_k_t`(L684), `set_mu_f`(L687), `set_smooth_friction`(L691), `set_s0`(L695), `set_smooth_blend`(L699), `multiply_weights`(L708), `reset_for_new_batch`(L713), `reset_multipliers`(L726)

### `src/physics/contact/contact_normal_alm.py`
- def `_to_tf` (L57)
- def `softplus_neg` (L64)
- def `fb_residual` (L81)
- class `NormalALMConfig` (L87): (no methods)
- class `NormalContactALM` (L98): `__init__`(L114), `build_from_numpy`(L145), `build_from_cat`(L219), `_auto_orient_normals`(L244), `_gap`(L263), `energy`(L294), `update_multipliers`(L352), `residual`(L389), `set_beta`(L446), `set_mu_n`(L450), `multiply_weights`(L454), `_compute_effective_pressure`(L468), `effective_normal_pressure`(L475), `reset_for_new_batch`(L481), `reset_multipliers`(L495)
- def `tfp_median` (L506)
- def `_ensure_2d` (L524)

### `src/physics/contact/contact_operator.py`
- class `ContactOperatorConfig` (L45): (no methods)
- class `ContactOperator` (L63): `__init__`(L74), `_friction_active`(L101), `build_from_cat`(L118), `reset_for_new_batch`(L172), `reset_multipliers`(L181), `energy`(L193), `residual`(L234), `update_multipliers`(L260), `last_sample_metrics`(L286), `last_meta`(L309), `snapshot_stage_state`(L316), `restore_stage_state`(L322), `last_friction_slip`(L327), `set_beta`(L335), `set_mu_n`(L339), `set_mu_t`(L343), `set_k_t`(L347), `set_mu_f`(L351), `multiply_weights`(L356), `N`(L369), `built`(L373)

### `src/physics/elasticity_config.py`
- class `ElasticityConfig` (L13): (no methods)

### `src/physics/elasticity_residual.py`
- class `ElasticityResidual` (L23): `__init__`(L24), `set_sample_indices`(L85), `set_sample_metrics_cache_enabled`(L92), `last_sample_metrics`(L97), `_cache_metrics`(L100), `_select_points`(L113), `_eval_u_on_nodes`(L123), `_compute_strain`(L126), `_compute_strain_reverse_mode`(L131), `_compute_strain_forward_mode`(L167), `_sigma_from_eps`(L221), `energy`(L232), `residual_cache`(L266)

### `src/physics/material_lib.py`
- def `lame_from_E_nu` (L41)
- def `isotropic_C_6x6` (L48)
- class `MaterialSpec` (L73): (no methods)
- class `MaterialLibrary` (L79): `__init__`(L91), `tags`(L124), `num_materials`(L127), `encode_tags`(L130), `id_of`(L141), `C_table_np`(L144), `C_table_tf`(L148), `summary`(L157)

### `src/physics/tightening_model.py`
- class `NutSpec` (L28): (no methods)
- class `TighteningConfig` (L36): (no methods)
- class `NutSampleData` (L44): (no methods)
- def `_sorted_node_ids` (L54)
- def `_map_node_ids_to_idx` (L58)
- def `_compute_area_weights` (L74)
- def `_normalize_axis` (L88)
- def `_auto_axis_from_nodes` (L98)
- class `NutTighteningPenalty` (L114): `__init__`(L115), `build_from_specs`(L119), `_u_fn_chunked`(L165), `_angle_to_rad`(L179), `_rotation_displacement`(L185), `energy`(L207), `residual`(L271)

### `src/train/__init__.py`
- (no top-level class/def)

### `src/train/loss_weights.py`
- def `_to_float` (L40)
- class `LossWeightState` (L51): `from_config`(L124), `as_dict`(L199)
- def `update_loss_weights` (L208)
- def `combine_loss` (L463)

### `src/train/uncertainty_calibration.py`
- def `calibrate_sigma_by_residual` (L10)

### `src/viz/__init__.py`
- (no top-level class/def)

### `src/viz/mirror_viz.py`
- def `_coerce_params_for_forward` (L42)
- def `_eval_displacement_batched` (L80)
- def `_with_new_stem` (L106)
- def `_eval_surface_or_assembly` (L116)
- def `_refine_surface_samples` (L171)
- def `_build_vertex_adjacency` (L236)
- def `_interpolate_displacement_on_refined` (L245)
- def `_smooth_scalar_on_tri_mesh` (L274)
- class `BlankRegionDiagnostics` (L324): `summary_lines`(L339), `primary_cause`(L352)
- def `_convex_hull_area` (L368)
- def `_triangle_area_sum` (L403)
- def `_collect_boundary_loops` (L414)
- def `_loop_area` (L471)
- def `_diagnose_blank_regions` (L483)
- def `_mask_tris_with_loops` (L545)
- def `_fit_rigid_transform` (L605)
- def `_remove_rigid_body_motion` (L633)
- def `_apply_rigid_correction` (L656)
- def `_fit_plane_basis` (L671)
- def `_unique_nodes_from_tris` (L695)
- def `_export_surface_mesh` (L707)
- def `_project_to_plane` (L736)
- def `plot_mirror_deflection` (L750)
- def `plot_mirror_deflection_by_name` (L1328)

### `test_forward_mode_normalize_inputs.py`
- class `ForwardModeNormalizeInputsTests` (L21): `test_normalize_inputs_supports_forward_accumulator_in_tf_function`(L22)

### `test_innovation_physics_losses.py`
- class `InnovationPhysicsLossTests` (L25): `test_incremental_ed_penalty_positive_when_violated`(L26), `test_incremental_ed_penalty_zero_when_satisfied`(L37), `test_friction_bipotential_term_exposed_in_stats`(L48), `test_uncertainty_proxy_sigma_shape_and_positive`(L81)

### `test_model_innovation_hooks.py`
- def `_make_minimal_asm` (L27)
- class `InnovationHookTests` (L49): `test_displacement_model_supports_finite_spectral_semantic_and_uncertainty`(L50), `test_build_node_semantic_features_shape_and_value_range`(L90), `test_residual_driven_sigma_calibration_is_monotonic`(L106), `test_graph_mode_without_prebuilt_graph_uses_dynamic_knn`(L119), `test_sample_level_adaptive_depth_routes_easy_and_hard_samples`(L163), `test_pointwise_forward_bypasses_graph_path`(L222)

### `test_performance_sync_guards.py`
- class `_MatLib` (L26): (no methods)
- class `PerformanceSyncGuardTests` (L30): `test_prepare_config_reads_rar_flags_from_yaml`(L31), `test_elasticity_metrics_cache_can_be_disabled`(L57)

### `test_trainer_optimization_hooks.py`
- class `_OptWithAggregateArg` (L26): `apply_gradients`(L27)
- class `_OptNoAggregateArg` (L32): `apply_gradients`(L33)
- class `TrainerOptimizationHookTests` (L38): `test_savedmodel_module_run_disables_autograph`(L39), `test_contact_route_update_interval_gate`(L59), `test_step_scalar_collection_uses_log_and_early_exit_intervals`(L69), `test_detect_apply_gradients_kwargs_for_supported_optimizer`(L84), `test_detect_apply_gradients_kwargs_for_plain_optimizer`(L88), `test_static_weight_vector_cache_for_non_adaptive_mode`(L92), `test_format_energy_summary_is_skipped_when_step_bar_disabled`(L112), `test_volume_sampling_falls_back_to_uniform_before_rar_cache_ready`(L129), `test_early_exit_triggers_after_nonfinite_streak`(L154), `test_early_exit_triggers_on_sustained_divergence`(L178), `test_contact_residual_route_metric_and_hint_push`(L205), `test_contact_multiplier_updates_are_plain_python_methods`(L232)

### `tools/audit_preload.py`
- def `_load_yaml` (L28)
- def `_resolve_surface_key` (L36)
- def `_parse_vec` (L55)
- def `_find_latest_saved_model` (L62)
- class `BoltAudit` (L73): (no methods)
- def `_audit_preload_geometry` (L82)
- def `_print_geometry_report` (L107)
- def `_make_u_fn_from_saved_model` (L129)
- def `main` (L152)

### `tools/bolt_region_metrics.py`
- def `_load_yaml` (L31)
- def `_resolve_surface_key` (L39)
- class `DeflectionData` (L59): (no methods)
- def `_load_deflection_txt` (L67)
- class `BoltCenter` (L92): (no methods)
- def `_load_bolt_centers` (L97)
- def `_region_stats` (L136)
- def `main` (L157)

### `tools/cdb_to_json.py`
- def `_parse_fixed_width` (L18)
- def `_safe_int` (L31)
- def `_safe_float` (L46)
- def `_expand_range_stream` (L58)
- def `_parse_cdb` (L72)
- def `main` (L233)

### `tools/compare_deflections.py`
- class `DeflectionData` (L25): (no methods)
- def `_load_txt` (L34)
- def `_ensure_same_nodes` (L65)
- def `_pearson` (L75)
- def `_summarize_one` (L84)
- def `_pair_metrics` (L101)
- def `_save_scatter` (L142)
- def `_iter_txt_paths` (L157)
- def `main` (L168)

### `tools/export_from_ckpt.py`
- def `_default_export_dir` (L21)
- def `main` (L26)

### `tools/visualize_contact_tightening.py`
- (no top-level class/def)

### `tools/visualize_contact_tightening_plotly.py`
- def `_load_config` (L37)
- def `_resolve_mesh_path` (L42)
- def `_load_asm` (L56)
- def `_normalize_contact_pairs` (L63)
- def `_normalize_axis` (L90)
- def `_auto_axis_from_nodes` (L103)
- def `_rotate_points` (L115)
- def `_tri_surface_to_mesh` (L128)
- def `_lighten` (L150)
- def `_rgb` (L157)
- def `_nut_specs_from_cfg` (L162)
- def `_tighten_angles_from_cfg` (L180)
- def `_order_from_cfg` (L189)
- def `_build_frame_angles` (L203)
- def `_color_cycle` (L222)
- def `_make_part_color_map` (L237)
- def `_build_vertex_neighbors` (L250)
- def `_laplacian_step` (L260)
- def `_taubin_smooth` (L270)
- def `_bounds_from_verts` (L287)
- def `_merge_bounds` (L293)
- def `_pad_bounds` (L306)
- def `_hover_template` (L313)
- def `_clip_faces_to_cylindrical_bounds` (L339)
- def `_mesh_trace` (L366)
- def `main` (L399)

### `tools/viz_saved_model.py`
- def `_load_yaml` (L29)
- def `_find_latest_saved_model` (L37)
- def `_parse_vec` (L47)
- def `_resolve_path` (L54)
- def `_load_saved_u_fn` (L61)
- def `_default_cases` (L93)
- def `main` (L117)
