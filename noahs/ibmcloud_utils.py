import requests



def cos_kwargs(ibm_creds, resiliency = 'regional', region = 'us-east', 
tier = 'public', subregion=None):
    '''
    Description: map the dict of credentials provided by ibm to the 
    kwargs requested by the COS session constructor.

    Issue: is it possible to use the API for find the exact endpoint 
    URL..? like building the constructor without an endpoint and 
    seeing if it tells you what the reight region and resiliancy, etc 
    should be????
    '''

    # set endpoint url config
    if not subregion:
        subregion = region # only different from region for "cross-region" reciliancy
    print('resiliency: ', resiliency, '\n', 'region: ', region, '\n', 'tier: ', tier, '\n', sep='')
    if subregion != region: print('subregion:', subregion)

    # retrieve endpoint url
    r = requests.get(ibm_creds['endpoints'])
    endpoint_url = r.json()['service-endpoints'][resiliency][region][tier][subregion]
    # alternatively set the enpoint url manually!
    #endpoint_url = <url copied from COS GUI>

    ## build authentification dict of keyword args 
    credentials = {
    # instance specific
    "ibm_api_key_id": ibm_creds['apikey'],
    "ibm_service_instance_id": ibm_creds["resource_instance_id"],
    
    # constants
    "ibm_auth_endpoint" : "https://iam.cloud.ibm.com/identity/token",
    "endpoint_url" : "https://" + endpoint_url
    }

    if 'cos_hmac_keys' in ibm_creds:
        credentials.update({
        #hmac keys
        'aws_access_key_id' : ibm_creds['cos_hmac_keys']["access_key_id"],
        'aws_secret_access_key' : ibm_creds['cos_hmac_keys']["secret_access_key"],
        })

    return credentials