from textloader import code2string, string2code,id2lettre
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  TODO:  Ce fichier contient les différentes fonction de génération


def generation_argmax(rnn, X, eos, start="", maxlen=200):
    X = torch.unsqueeze(string2code(start), 1)
    with torch.no_grad():
        h0 = torch.zeros(1, rnn.dim_latent)
        ht1 = rnn.forward(X.to(device), h0.to(device))
        dt1 = rnn.decode(torch.unsqueeze(ht1[-1],0))
        yt1 = F.softmax(dt1, dim=2).argmax(dim=2)
        ht1 = ht1[-1]
        logits = dt1[0,0,yt1]
        i = 1
        while yt1[-1] != eos and i < maxlen:
            #print(torch.unsqueeze(yt1[-1],0).shape)
            ht1 = rnn.one_step(torch.unsqueeze(yt1[-1],0), ht1)
            dt1 = rnn.decode(torch.unsqueeze(ht1,0))
            y = F.softmax(dt1, dim=2).argmax(dim=2)
            yt1 = torch.cat((yt1, y))
            logits = torch.cat((logits, dt1[0,0,yt1[-1]].unsqueeze(0)))
            i+=1

        X.cpu()
        h0.cpu()
        ht1.cpu()
        yt1.cpu()
        y.cpu()
        dt1.cpu()
        logits.cpu()

    sequence = code2string(yt1)
             
    return sequence, logits



def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    X = torch.unsqueeze(string2code(start), 1)
    with torch.no_grad():
            h0 = torch.zeros(1, rnn.dim_latent)
            ht1 = rnn.forward(X.to(device), h0.to(device))
            dt1 = rnn.decode(torch.unsqueeze(ht1[-1],0))
            yt1 = torch.multinomial(torch.flatten(F.softmax(dt1, dim=2)), 1, replacement=True)

            ht1 = ht1[-1]
            logits = dt1[0,0,yt1]
            i = 1
            while yt1[-1] != eos and i < maxlen:
                #print(torch.unsqueeze(yt1[-1],0).shape)
                ht1 = rnn.one_step(torch.unsqueeze(yt1[-1],0), ht1)
                dt1 = rnn.decode(torch.unsqueeze(ht1,0))
                y = torch.multinomial(torch.flatten(F.softmax(dt1, dim=2)), 1, replacement=True)
                yt1 = torch.cat((yt1, y))
                logits = torch.cat((logits, dt1[0,0,yt1[-1]].unsqueeze(0)))
                i+=1

            X.cpu()
            h0.cpu()
            ht1.cpu()
            yt1.cpu()
            y.cpu()
            dt1.cpu()
            logits.cpu()

    sequence = code2string(yt1)
             
    return sequence, logits

    
def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200, nucleus=False):
    #  TODO:  Implémentez le beam Search
    X = torch.unsqueeze(string2code(start), 1)
    with torch.no_grad():
            h0 = torch.zeros(1, rnn.dim_latent)
            ht1 = rnn.forward(X.to(device), h0.to(device))
            dt1 = rnn.decode(torch.unsqueeze(ht1[-1],0))
            if nucleus:
                # sampling k label with nucleus
                log_proba_t1, yt1 = p_nucleus(torch.flatten(dt1), k)
            else:
                log_proba_t1, yt1 = torch.topk(torch.flatten(F.log_softmax(dt1, dim=2)), k)

            #sequence start and serves to save k best sequences
            ht1 = ht1[-1].expand(k,ht1.shape[2])
            log_proba_t1 = log_proba_t1.unsqueeze(1)
            yt1 = yt1.unsqueeze(1)

            seq_len = 1
            h_tmp = torch.zeros(k*k, ht1.shape[1], device=device)
            y_tmp = torch.zeros(k*k, 1, device=device)
            log_proba_tmp = torch.zeros(k*k, 1, device=device)
            while seq_len < maxlen:
                for rank in range(k):
                    if yt1[rank, seq_len-1] != eos and yt1[rank, seq_len-1] != 0:
                        h = rnn.one_step(yt1[rank, seq_len-1].unsqueeze(0), ht1[rank].unsqueeze(0))
                        dt1 = rnn.decode(torch.unsqueeze(h,0))
                        if nucleus:
                            # sampling k label with nucleus
                            log_proba, y = p_nucleus(torch.flatten(dt1), k)

                        else:
                            log_proba, y = torch.topk(torch.flatten(F.log_softmax(dt1, dim=2)), k)
                        h_tmp[rank*k:rank*k+k] = h.expand(k,h.shape[1])
                        y_tmp[rank*k:rank*k+k] = y.view(-1,1)
                        log_proba_tmp[rank*k:rank*k+k] = log_proba.view(-1,1) + log_proba_t1[rank]

                    else:
                        log_proba_tmp[rank*k:rank*k+k] = log_proba_t1[rank].clone()

                #select k best sequence
                _, k_best_ind = torch.topk(torch.flatten(log_proba_tmp), k)
                #save k best
                ht1 = h_tmp[k_best_ind].clone()
                log_proba_t1 = log_proba_tmp[k_best_ind].clone()
                yt1 = torch.cat((yt1, y_tmp[k_best_ind].clone()), dim=1)
                
                # reset tmp variables
                h_tmp *=0 
                y_tmp *=0
                log_proba_tmp *=0

                seq_len+=1

            X.cpu()
            h0.cpu()
            ht1.cpu()
            yt1.cpu()
            yt1.cpu()
            log_proba_t1.cpu()
            dt1.cpu()
            h_tmp.cpu()
            y_tmp.cpu()
            log_proba_tmp.cpu()

    #select best sequence
    print(log_proba_t1)
    ind = torch.flatten(log_proba_t1).argmax()
    best_seq = yt1[ind]
    sequence = code2string(best_seq)
             
    return sequence



# p_nucleus
def p_nucleus(logits, k: int):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        k (int): [description]
    """

    #getting k best symbols
    proba_k, y_k = torch.topk(F.softmax(logits), k)
    proba_k = proba_k / proba_k.sum()

    #generate k symbols following best k distribution
    ind = torch.multinomial(proba_k, k, replacement=True)
    # y generated
    y_k = y_k[ind]
    proba_k = torch.log(proba_k[ind])
    #proba_k = F.log_softmax(logits)[y_k]
    
    return proba_k, y_k
