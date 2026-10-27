function st = st_init()
% ST_INIT  Patient-reference state (reset at the start of every record).
st = struct('rr_ref', 0, 'qrs_ref', 0, 'p_ref', 0, 't_ref', 0, 'r_ref', 0, ...
            'rt_ref', 0, 'pol_cnt', 0, 'n_rr', 0, 'n_mor', 0);
end
