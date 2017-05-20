function [ v_norm ] = normalize( v )
%NORMALIZE the given vektor v
    v_norm = (v - mean(v))/std(v);
end

