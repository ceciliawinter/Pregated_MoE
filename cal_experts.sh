for i in {8,64,128}
do
    for j in {1,2,4,8,16,32,128,256,512}
    do
        for k in Pre-gated GPU-only DeepSpeed SE-MoE
        do
            if [ -e logs/switch-base-${i}_${k}_${j}_0_0_0.log ]; then
                grep "^permuted_experts_" logs/switch-base-${i}_${k}_${j}_0_0_0.log > expert_log/experts_e${i}_${k}_b${j}.txt
            fi
        done
    done

done
