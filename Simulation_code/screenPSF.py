import numpy as np

def screenPSF(N, NA, F0, screen):
    # Extract intensities
    val = screen[1, :]
    
    # Calculate the speckle + random phase at the wavelength with the peak intensity
    y = np.argmax(val)  # Find the peak position
    lambda_m = screen[0, y]
    value_m = screen[1, y]
    F_cut_m = 2 * np.pi * NA / lambda_m
    F_CutOff_ratio_m = F_cut_m / F0
    
    # Create meshgrid for row and column
    r, c = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    R2 = r**2 + c**2
    
    # Generate amplitude mask and random phase for peak wavelength
    Amp_m = (R2 < (N * F_CutOff_ratio_m)**2).astype(float)
    ScatPhase_m = 200 * np.pi * np.random.rand(*Amp_m.shape)
    speckle_m = np.fft.ifftshift(np.fft.ifft2(Amp_m * np.exp(1j * ScatPhase_m)))
    speckle_m = np.abs(speckle_m)**2
    speckle_m /= np.sum(speckle_m)
    PSF_m = speckle_m
    
    # Remove the peak intensity wavelength from screen
    screen = np.delete(screen, y, axis=1)
    
    # Initialize the PSF array
    PSF = []

    # Generate speckle at other wavelengths
    for i in range(screen.shape[1]):
        lambda_ = screen[0, i]
        value = screen[1, i]
        F_cut = 2 * np.pi * NA / lambda_
        F_CutOff_ratio = F_cut / F0
        Amp = (R2 < (N * F_CutOff_ratio)**2).astype(float)
        Amp *= (value / value_m)  # Scale amplitude by intensity ratio
        ScatPhase = ScatPhase_m * (lambda_m / lambda_)  # Scale phase by wavelength ratio
        speckle = np.fft.ifftshift(np.fft.ifft2(Amp * np.exp(1j * ScatPhase)))
        speckle = np.abs(speckle)**2
        speckle /= np.sum(speckle)
        PSF.append(speckle)
    
    # Sum all speckles at different wavelengths
    PSF = np.stack(PSF, axis=2)
    Speckle = np.sum(PSF, axis=2) + PSF_m
    
    return Speckle
