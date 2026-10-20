"""Tail mapping parity, stale padding and address stability for the gfx908 fusion."""
import pytest
import torch
from vllm.v1.attention.backends.mla import indexer


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires GPU')
@pytest.mark.parametrize('lengths', [(1,), (1,1), (3,4,5), (63,64,65), (639,640,641), (1280,1280)])
def test_tail_mapping_strides_padding_and_replay(monkeypatch, lengths):
    device = 'cuda'
    total = sum(lengths)
    # Keep allocation larger than both captured and later actual batch lengths.
    capacity = total+513
    wide = torch.full((len(lengths),8),-99,dtype=torch.int32,device=device)
    table = wide[:,2:5]
    table[:,0] = torch.arange(len(lengths),device=device,dtype=torch.int32)*7+13
    qsl_cpu=[0]
    for n in lengths: qsl_cpu.append(qsl_cpu[-1]+n)
    qsl=torch.tensor(qsl_cpu,dtype=torch.int32,device=device)
    pos_wide=torch.arange(2*(total+16),dtype=torch.int64,device=device)
    positions=pos_wide[::2]
    slots=torch.full((total+16,),999,dtype=torch.int64,device=device)
    output=torch.full((capacity,),888,dtype=torch.int64,device=device)
    ptr=output.data_ptr()
    def run(n):
        return indexer.compute_kpool_tail_slot_mapping(slots,table,qsl,positions,n,len(lengths),4,output)
    monkeypatch.setattr(indexer,'USE_GLM53_TAIL_FUSION',False)
    run(total); reference=output.clone()
    monkeypatch.setattr(indexer,'USE_GLM53_TAIL_FUSION',True)
    actual=run(total)
    torch.testing.assert_close(output,reference,rtol=0,atol=0)
    assert actual.data_ptr()==ptr
    # Kernel writes every persistent padding row, not just the returned view.
    output.fill_(123456)
    run(0)
    assert bool((output==-1).all())
    # Capture executes with persistent output/input addresses; repeated replay
    # must read changed block-table and position contents rather than constants.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): run(total)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run(total)
    table[:,0]+=5
    positions[:total]+=19
    graph.replay()
    captured=output.clone()
    monkeypatch.setattr(indexer,'USE_GLM53_TAIL_FUSION',False)
    run(total)
    torch.testing.assert_close(captured,output,rtol=0,atol=0)
    assert output.data_ptr()==ptr
